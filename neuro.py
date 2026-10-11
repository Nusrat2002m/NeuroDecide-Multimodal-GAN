!pip install -q -U kagglehub

import os, glob, random, zipfile, warnings, traceback
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import (confusion_matrix, roc_curve, auc, precision_score,
                              recall_score, f1_score, accuracy_score, roc_auc_score)
import tensorflow as tf
from tensorflow.keras import layers, models

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED)

OUT_DIR = "/content/NeuroDecideGAN_outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

DPI = 400
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 13, 'font.family': 'DejaVu Sans',
    'axes.titlesize': 15, 'axes.titleweight': 'bold',
    'axes.labelsize': 13, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 11, 'figure.autolayout': True,
    'axes.edgecolor': '#333333', 'axes.linewidth': 1.2,
    'grid.alpha': 0.3
})

FIG_COUNT = [0]
def savefig(name):
    FIG_COUNT[0] += 1
    try:
        path = os.path.join(FIG_DIR, f"{FIG_COUNT[0]:02d}_{name}.png")
        plt.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
        print(f"[Figure saved] {os.path.basename(path)}")
    except Exception as e:
        print(f"[WARN] Could not save figure '{name}': {e}")
    finally:
        plt.close('all')

def safe_run(stage_name, fn, *args, **kwargs):
    print(f"\n>>> {stage_name}")
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"[WARN] Stage '{stage_name}' failed: {e}")
        traceback.print_exc()
        return None

# ============================================================
# 1. DOWNLOAD DATASETS
# ============================================================
import kagglehub

def download(slug):
    return kagglehub.dataset_download(slug)

vision_path   = safe_run("Downloading vision dataset",   download, "uraninjo/augmented-alzheimer-mri-dataset")
clinical_path = safe_run("Downloading clinical dataset",  download, "jboysen/mri-and-alzheimers")
eeg_path      = safe_run("Downloading EEG dataset",       download, "birdy654/eeg-brainwave-dataset-mental-state")


def find_class_dir(root, min_classes=2, prefer_keyword='train'):
    best, best_score = None, -1
    for dirpath, dirnames, _ in os.walk(root):
        subdirs_imgs = []
        for d in dirnames:
            sub = os.path.join(dirpath, d)
            try:
                imgs = [f for f in os.listdir(sub) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            except Exception:
                imgs = []
            if imgs:
                subdirs_imgs.append((d, len(imgs)))
        if len(subdirs_imgs) >= min_classes:
            score = sum(c for _, c in subdirs_imgs)
            if prefer_keyword in dirpath.lower():
                score += 10**6
            if score > best_score:
                best_score, best = score, dirpath
    return best

def load_vision(root, img_size=64, max_per_class=250):
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    class_dir = find_class_dir(root)
    if not class_dir:
        raise ValueError("No class directory found in vision dataset.")
    classes = sorted([d for d in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, d))])
    X, y = [], []
    for ci, c in enumerate(classes):
        cdir = os.path.join(class_dir, c)
        files = [f for f in os.listdir(cdir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(files)
        for f in files[:max_per_class]:
            try:
                arr = img_to_array(load_img(os.path.join(cdir, f), target_size=(img_size, img_size))) / 255.0
                X.append(arr); y.append(ci)
            except Exception:
                continue
    X, y = np.array(X, dtype=np.float32), np.array(y)
    if len(X) < 10:
        raise ValueError("Too few valid vision images loaded.")
    # FIX: belt-and-suspenders safety net — corrupt image files can occasionally decode
    # to NaN/Inf pixel values; scrub them so a single bad file can't poison training.
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    return X, y, classes

def synthetic_vision(n_classes=4, n_per_class=120, size=64):
    X = np.random.rand(n_classes * n_per_class, size, size, 3).astype(np.float32)
    y = np.repeat(np.arange(n_classes), n_per_class)
    classes = [f"SyntheticClass{i}" for i in range(n_classes)]
    return X, y, classes

def find_csv(root, keyword=None):
    csvs = glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    if keyword:
        kw = [c for c in csvs if keyword.lower() in os.path.basename(c).lower()]
        if kw:
            csvs = kw
    if not csvs:
        raise ValueError("No CSV found.")
    csvs.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return csvs[0]

def split_features_label(df, priority_keywords=('group', 'diagnos', 'label', 'class', 'condition', 'state')):
    df = df.copy()
    label_col = None
    for kw in priority_keywords:
        for c in df.columns:
            if kw in c.lower() and 2 <= df[c].nunique() <= 10:
                label_col = c; break
        if label_col:
            break
    if label_col is None:
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        cand = [c for c in obj_cols if 2 <= df[c].nunique() <= 10]
        label_col = cand[0] if cand else (obj_cols[0] if obj_cols else df.columns[-1])
    y_raw = df[label_col].astype(str)
    X = df.drop(columns=[label_col]).apply(pd.to_numeric, errors='coerce')
    X = X.select_dtypes(include=[np.number])

    # FIX (root cause of the NaN crash): columns that are entirely non-numeric before
    # coercion (e.g. a Subject/MRI ID column like "OAS2_0001") become ALL-NaN after
    # pd.to_numeric(errors='coerce'). X.fillna(X.median()) does NOT fix an all-NaN
    # column, because the median of an all-NaN column is itself NaN — so NaN silently
    # passed through the scaler into the encoder and corrupted every downstream loss
    # that touched the clinical branch (clin_loss, disc_loss, adv_loss all went NaN
    # from step 1). Drop those columns BEFORE computing the median fill.
    n_before = X.shape[1]
    X = X.dropna(axis=1, how='all')
    n_after = X.shape[1]
    if n_after < n_before:
        print(f"[INFO] split_features_label: dropped {n_before - n_after} all-NaN column(s) "
              f"after numeric coercion (likely ID/text columns).")

    if X.shape[1] == 0:
        raise ValueError("No numeric features remain after label extraction.")

    X = X.fillna(X.median())
    # FIX: final hard safety net — guarantees no NaN/Inf can ever leave this function,
    # even if some future dataset variant has an edge case the logic above doesn't catch.
    X = X.fillna(0.0)
    X_arr = np.nan_to_num(X.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return X_arr, y, list(le.classes_), label_col

def synthetic_tabular(n=300, n_features=15, classes=('A', 'B', 'C')):
    X = np.random.randn(n, n_features).astype(np.float32)
    y = np.random.randint(0, len(classes), n)
    return X, y, list(classes)

# ---- Load Vision ----
result = safe_run("Loading vision images", load_vision, vision_path) if vision_path else None
if result is None:
    print("[INFO] Using synthetic vision fallback.")
    X_vis, y_vis, vision_classes = synthetic_vision()
else:
    X_vis, y_vis, vision_classes = result
IMG_SIZE = X_vis.shape[1]

# ---- Load EEG ----
eeg_df = None
if eeg_path:
    f = safe_run("Locating EEG csv", find_csv, eeg_path)
    if f:
        eeg_df = safe_run("Reading EEG csv", pd.read_csv, f)
if eeg_df is None or len(eeg_df) < 10:
    print("[INFO] Using synthetic EEG fallback.")
    eeg_X, eeg_y, eeg_classes = synthetic_tabular(classes=('RELAXED', 'NEUTRAL', 'CONCENTRATING'))
else:
    res = safe_run("Parsing EEG features/labels", split_features_label, eeg_df)
    eeg_X, eeg_y, eeg_classes = (res[0], res[1], res[2]) if res else synthetic_tabular(classes=('RELAXED', 'NEUTRAL', 'CONCENTRATING'))

# ---- Load Clinical ----
clinical_df = None
if clinical_path:
    f = safe_run("Locating clinical csv", find_csv, clinical_path, "long")
    if not f:
        f = safe_run("Locating clinical csv (any)", find_csv, clinical_path)
    if f:
        clinical_df = safe_run("Reading clinical csv", pd.read_csv, f)
if clinical_df is None or len(clinical_df) < 10:
    print("[INFO] Using synthetic clinical fallback.")
    clin_X, clin_y, clin_classes = synthetic_tabular(classes=('Demented', 'Nondemented', 'Converted'))
else:
    res = safe_run("Parsing clinical features/labels", split_features_label, clinical_df)
    clin_X, clin_y, clin_classes = (res[0], res[1], res[2]) if res else synthetic_tabular(classes=('Demented', 'Nondemented', 'Converted'))

print(f"\nVision: {X_vis.shape}, classes={vision_classes}")
print(f"EEG: {eeg_X.shape}, classes={eeg_classes}")
print(f"Clinical: {clin_X.shape}, classes={clin_classes}")

# FIX: explicit data-validation checkpoint. If any branch still contains NaN/Inf at this
# point, fail loudly HERE with a clear message, instead of three training steps later with
# a cryptic "NaN loss" that gives no clue which array caused it.
def assert_clean(name, arr):
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    status = "OK" if (n_nan == 0 and n_inf == 0) else "PROBLEM"
    print(f"[Data check] {name}: shape={arr.shape}, NaN={n_nan}, Inf={n_inf} -> {status}")
    if n_nan or n_inf:
        raise ValueError(f"{name} contains NaN/Inf after preprocessing — fix the loader before training.")

assert_clean("X_vis", X_vis)
assert_clean("eeg_X", eeg_X)
assert_clean("clin_X", clin_X)

# ============================================================
# 3. TRAIN/TEST SPLITS  (scalers fit on train split only)
# ============================================================
def safe_split(X, y):
    try:
        return train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    except Exception:
        return train_test_split(X, y, test_size=0.2, random_state=SEED)

X_vis_tr, X_vis_te, y_vis_tr, y_vis_te = safe_split(X_vis, y_vis)
eeg_X_tr, eeg_X_te, eeg_y_tr, eeg_y_te = safe_split(eeg_X, eeg_y)
clin_X_tr, clin_X_te, clin_y_tr, clin_y_te = safe_split(clin_X, clin_y)

# Fit StandardScaler on the TRAIN split only, then transform train and test with it
# (avoids test-set information leaking into the scaler's mean/variance).
eeg_scaler = StandardScaler().fit(eeg_X_tr)
eeg_X_tr = eeg_scaler.transform(eeg_X_tr)
eeg_X_te = eeg_scaler.transform(eeg_X_te)

clin_scaler = StandardScaler().fit(clin_X_tr)
clin_X_tr = clin_scaler.transform(clin_X_tr)
clin_X_te = clin_scaler.transform(clin_X_te)

# Full-array scaled versions, for visualization figures ONLY (never touches training/metrics).
eeg_X_scaled_full = eeg_scaler.transform(eeg_X)
clin_X_scaled_full = clin_scaler.transform(clin_X)

# FIX: re-validate after scaling too — a zero-variance column is handled safely by
# sklearn, but this check costs nothing and catches any future regression immediately.
assert_clean("eeg_X_tr (scaled)", eeg_X_tr)
assert_clean("clin_X_tr (scaled)", clin_X_tr)

N_VIS, N_EEG, N_CLIN = len(set(y_vis)), len(set(eeg_y)), len(set(clin_y))

# ============================================================
# 4. NEURODECIDE-GAN ARCHITECTURE
# ============================================================
EMB_DIM = 64

def build_vision_encoder():
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(EMB_DIM, activation='relu')(x)
    return models.Model(inp, x, name='Vision_Encoder')

def build_tabular_encoder(input_dim, name):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(EMB_DIM, activation='relu')(x)
    return models.Model(inp, x, name=name)

def build_classifier(num_classes, name):
    inp = layers.Input(shape=(EMB_DIM,))
    x = layers.Dense(32, activation='relu')(inp)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inp, out, name=name)

def build_discriminator():
    inp = layers.Input(shape=(EMB_DIM,))
    x = layers.Dense(32, activation='relu')(inp)
    out = layers.Dense(3, activation='softmax')(x)
    return models.Model(inp, out, name='Modality_Discriminator')

vis_enc = build_vision_encoder()
eeg_enc = build_tabular_encoder(eeg_X.shape[1], 'EEG_Encoder')
clin_enc = build_tabular_encoder(clin_X.shape[1], 'Clinical_Encoder')
vis_clf = build_classifier(N_VIS, 'Vision_Classifier')
eeg_clf = build_classifier(N_EEG, 'EEG_Classifier')
clin_clf = build_classifier(N_CLIN, 'Clinical_Classifier')
disc = build_discriminator()

emb_before = {
    'Vision': vis_enc.predict(X_vis_te, verbose=0),
    'EEG': eeg_enc.predict(eeg_X_te, verbose=0),
    'Clinical': clin_enc.predict(clin_X_te, verbose=0),
}

# ============================================================
# 5. ADVERSARIAL TRAINING LOOP
# ============================================================
EPOCHS, STEPS_PER_EPOCH, BATCH, LAMBDA_ADV = 15, 40, 16, 0.3

# Gradient clipping on both optimizers — kept from the previous patch as a safety net
# against exploding gradients, independent of the NaN-column bug fixed above.
opt_enc = tf.keras.optimizers.Adam(1e-3, clipnorm=1.0)
opt_disc = tf.keras.optimizers.Adam(1e-3, clipnorm=1.0)
cce = tf.keras.losses.SparseCategoricalCrossentropy()

history = {k: [] for k in ['vis_loss', 'eeg_loss', 'clin_loss', 'disc_loss', 'adv_loss', 'disc_acc']}

def sample_batch(X, y, n):
    idx = np.random.randint(0, len(X), size=n)
    return X[idx], y[idx]

def train_step():
    xv, yv = sample_batch(X_vis_tr, y_vis_tr, BATCH)
    xe, ye = sample_batch(eeg_X_tr, eeg_y_tr, BATCH)
    xc, yc = sample_batch(clin_X_tr, clin_y_tr, BATCH)

    with tf.GradientTape() as tape:
        ev = vis_enc(xv, training=False); ee = eeg_enc(xe, training=False); ec = clin_enc(xc, training=False)
        emb_all = tf.concat([ev, ee, ec], axis=0)
        mod_labels = tf.concat([tf.zeros(BATCH, tf.int32), tf.ones(BATCH, tf.int32), tf.fill([BATCH], 2)], axis=0)
        preds = disc(emb_all, training=True)
        d_loss = cce(mod_labels, preds)
    grads = tape.gradient(d_loss, disc.trainable_variables)
    opt_disc.apply_gradients(zip(grads, disc.trainable_variables))
    d_acc = np.mean(np.argmax(preds.numpy(), axis=1) == mod_labels.numpy())

    with tf.GradientTape() as tape:
        ev = vis_enc(xv, training=True); ee = eeg_enc(xe, training=True); ec = clin_enc(xc, training=True)
        pv = vis_clf(ev, training=True); pe = eeg_clf(ee, training=True); pc = clin_clf(ec, training=True)
        l_v, l_e, l_c = cce(yv, pv), cce(ye, pe), cce(yc, pc)
        emb_all2 = tf.concat([ev, ee, ec], axis=0)
        disc_preds2 = disc(emb_all2, training=False)
        uniform = tf.ones_like(disc_preds2) / 3.0
        adv_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(uniform, disc_preds2))
        total = l_v + l_e + l_c + LAMBDA_ADV * adv_loss
    trainables = (vis_enc.trainable_variables + eeg_enc.trainable_variables + clin_enc.trainable_variables +
                  vis_clf.trainable_variables + eeg_clf.trainable_variables + clin_clf.trainable_variables)
    grads = tape.gradient(total, trainables)
    opt_enc.apply_gradients(zip(grads, trainables))
    return l_v.numpy(), l_e.numpy(), l_c.numpy(), d_loss.numpy(), adv_loss.numpy(), d_acc

def run_training():
    for epoch in range(EPOCHS):
        buf = {k: [] for k in history}
        for step in range(STEPS_PER_EPOCH):
            lv, le, lc, dl, al, da = train_step()
            # NaN guard kept as a safety net — should no longer fire now that the root
            # cause (all-NaN clinical column) is fixed, but if it ever does, this stops
            # the run immediately instead of finishing on corrupted weights.
            if any(np.isnan(x) for x in [lv, le, lc, dl, al]):
                raise RuntimeError(
                    f"NaN loss detected at epoch {epoch+1}, step {step+1} "
                    f"(vis={lv}, eeg={le}, clin={lc}, disc={dl}, adv={al}). "
                    f"Stopping training instead of continuing on corrupted weights."
                )
            buf['vis_loss'].append(lv); buf['eeg_loss'].append(le); buf['clin_loss'].append(lc)
            buf['disc_loss'].append(dl); buf['adv_loss'].append(al); buf['disc_acc'].append(da)
        for k in history:
            history[k].append(float(np.mean(buf[k])))
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Vis {history['vis_loss'][-1]:.3f} | "
              f"EEG {history['eeg_loss'][-1]:.3f} | Clin {history['clin_loss'][-1]:.3f} | "
              f"Disc {history['disc_loss'][-1]:.3f} | Adv {history['adv_loss'][-1]:.3f} | "
              f"DiscAcc {history['disc_acc'][-1]:.3f}")

run_training()

emb_after = {
    'Vision': vis_enc.predict(X_vis_te, verbose=0),
    'EEG': eeg_enc.predict(eeg_X_te, verbose=0),
    'Clinical': clin_enc.predict(clin_X_te, verbose=0),
}

# ============================================================
# 6. EVALUATION + RESULTS TABLE
# ============================================================
def evaluate(encoder, clf, X_test, y_test):
    emb = encoder.predict(X_test, verbose=0)
    probs = clf.predict(emb, verbose=0)
    preds = np.argmax(probs, axis=1)
    acc = float(np.mean(preds == y_test))
    return probs, preds, acc

vis_probs, vis_preds, vis_acc = evaluate(vis_enc, vis_clf, X_vis_te, y_vis_te)
eeg_probs, eeg_preds, eeg_acc = evaluate(eeg_enc, eeg_clf, eeg_X_te, eeg_y_te)
clin_probs, clin_preds, clin_acc = evaluate(clin_enc, clin_clf, clin_X_te, clin_y_te)

def safe_auc(y_true, probs, n_classes):
    try:
        y_bin = np.eye(n_classes)[y_true]
        return roc_auc_score(y_bin, probs, average='macro', multi_class='ovr')
    except Exception:
        return np.nan

def build_results_row(name, y_true, y_pred, probs, n_classes):
    return {
        'Modality Branch': name,
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
        'Precision (macro)': round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'Recall (macro)': round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'F1-Score (macro)': round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'AUC (macro, OvR)': round(safe_auc(y_true, probs, n_classes), 4)
    }

results_rows = [
    build_results_row('Vision (MRI)', y_vis_te, vis_preds, vis_probs, N_VIS),
    build_results_row('Cognition (EEG)', eeg_y_te, eeg_preds, eeg_probs, N_EEG),
    build_results_row('Clinical Decision', clin_y_te, clin_preds, clin_probs, N_CLIN),
]
results_df = pd.DataFrame(results_rows)

overall_row = {
    'Modality Branch': 'NeuroDecide-GAN (Overall)',
    'Accuracy': round(results_df['Accuracy'].mean(), 4),
    'Precision (macro)': round(results_df['Precision (macro)'].mean(), 4),
    'Recall (macro)': round(results_df['Recall (macro)'].mean(), 4),
    'F1-Score (macro)': round(results_df['F1-Score (macro)'].mean(), 4),
    'AUC (macro, OvR)': round(results_df['AUC (macro, OvR)'].mean(), 4)
}
results_df = pd.concat([results_df, pd.DataFrame([overall_row])], ignore_index=True)

print("\n=== RESULTS TABLE ===")
print(results_df.to_string(index=False))

csv_path = os.path.join(TAB_DIR, "results_table.csv")
try:
    results_df.to_csv(csv_path, index=False)
    print(f"[Table saved] {csv_path}")
except Exception as e:
    print(f"[WARN] Could not save results CSV: {e}")

# ============================================================
# 7. FIGURES (400 DPI, publication style, high clarity)
# ============================================================

def fig_sample_images():
    n_show = min(8, len(X_vis))
    idx = np.random.choice(len(X_vis), n_show, replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, i in zip(axes.flat, idx):
        ax.imshow(X_vis[i]); ax.set_title(vision_classes[y_vis[i]], fontsize=12, fontweight='bold')
        ax.axis('off')
    fig.suptitle("Representative MRI Samples Across Classes", fontsize=17, fontweight='bold')
    savefig("sample_mri_images")
safe_run("Figure: sample MRI images", fig_sample_images)

def fig_class_distributions():
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    for ax, (labels, classes, title) in zip(axes, [
        (y_vis, vision_classes, "Vision (MRI) Class Distribution"),
        (eeg_y, eeg_classes, "Cognition (EEG) Class Distribution"),
        (clin_y, clin_classes, "Clinical Decision Class Distribution")]):
        counts = np.bincount(labels, minlength=len(classes))
        bars = ax.bar([str(c) for c in classes], counts, color=sns.color_palette("viridis", len(classes)),
                      edgecolor='black', linewidth=1.2)
        for b, c in zip(bars, counts):
            ax.text(b.get_x() + b.get_width()/2, c + max(counts)*0.02, str(c), ha='center', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold'); ax.set_ylabel("Sample Count")
        ax.tick_params(axis='x', rotation=25)
    plt.tight_layout()
    savefig("class_distributions")
safe_run("Figure: class distributions", fig_class_distributions)

def fig_architecture():
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.axis('off')
    boxes = {
        'Vision Encoder': (0.03, 0.72), 'EEG Encoder': (0.03, 0.45), 'Clinical Encoder': (0.03, 0.18),
        'Shared Latent\nSpace': (0.4, 0.45), 'Modality\nDiscriminator': (0.72, 0.72),
        'Vision Head': (0.72, 0.52), 'EEG Head': (0.72, 0.32), 'Clinical\nDecision Head': (0.72, 0.12)
    }
    colors = {'Vision Encoder': '#3B5998', 'EEG Encoder': '#2E8B57', 'Clinical Encoder': '#B22222',
              'Shared Latent\nSpace': '#6A5ACD', 'Modality\nDiscriminator': '#DAA520',
              'Vision Head': '#3B5998', 'EEG Head': '#2E8B57', 'Clinical\nDecision Head': '#B22222'}
    for name, (x, y) in boxes.items():
        w, h = 0.22, 0.14
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=colors[name], ec='black', linewidth=1.5, alpha=0.92)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, name, ha='center', va='center', fontsize=12, color='white', weight='bold')
    arrows = [('Vision Encoder', 'Shared Latent\nSpace'), ('EEG Encoder', 'Shared Latent\nSpace'),
              ('Clinical Encoder', 'Shared Latent\nSpace'), ('Shared Latent\nSpace', 'Modality\nDiscriminator'),
              ('Shared Latent\nSpace', 'Vision Head'), ('Shared Latent\nSpace', 'EEG Head'),
              ('Shared Latent\nSpace', 'Clinical\nDecision Head')]
    for a, b in arrows:
        xa, ya = boxes[a]; xb, yb = boxes[b]
        ax.add_patch(FancyArrowPatch((xa+0.22, ya+0.07), (xb, yb+0.07), arrowstyle='-|>', mutation_scale=22,
                                      color='#444444', linewidth=1.8))
    ax.set_title("NeuroDecide-GAN: Adversarial Multi-Modal Fusion Architecture", fontsize=17, fontweight='bold')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    savefig("architecture_diagram")
safe_run("Figure: architecture diagram", fig_architecture)

def fig_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
    ep = list(range(1, len(history['vis_loss']) + 1))
    for k, lbl, c in [('vis_loss', 'Vision', '#3B5998'), ('eeg_loss', 'EEG', '#2E8B57'), ('clin_loss', 'Clinical', '#B22222')]:
        axes[0].plot(ep, history[k], marker='o', markersize=6, linewidth=2.5, label=lbl, color=c)
    axes[0].set_title("Branch Classification Loss", fontsize=15, fontweight='bold')
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep, history['disc_loss'], marker='s', markersize=6, linewidth=2.5, label='Discriminator Loss', color='#B22222')
    axes[1].plot(ep, history['adv_loss'], marker='^', markersize=6, linewidth=2.5, label='Adversarial (Confusion) Loss', color='#3B5998')
    ax2 = axes[1].twinx()
    ax2.plot(ep, history['disc_acc'], marker='d', markersize=6, linewidth=2, linestyle='--', color='gray', label='Discriminator Accuracy')
    ax2.set_ylabel("Discriminator Accuracy")
    axes[1].set_title("Adversarial Alignment Dynamics", fontsize=15, fontweight='bold')
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    lines1, labels1 = axes[1].get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)
    axes[1].grid(alpha=0.3)
    savefig("training_curves")
safe_run("Figure: training curves", fig_training_curves)

def fig_tsne(embeddings, title, fname):
    all_emb, mod_labels = [], []
    for name, emb in embeddings.items():
        all_emb.append(emb); mod_labels += [name] * len(emb)
    all_emb = np.vstack(all_emb)
    perplex = min(30, max(5, len(all_emb)//10))
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=perplex, init='pca')
    proj = tsne.fit_transform(all_emb)
    df = pd.DataFrame({'x': proj[:, 0], 'y': proj[:, 1], 'Modality': mod_labels})
    plt.figure(figsize=(8.5, 7.5))
    sns.scatterplot(data=df, x='x', y='y', hue='Modality', palette='Set1', s=60, alpha=0.75, edgecolor='black', linewidth=0.4)
    plt.title(title, fontsize=15, fontweight='bold'); plt.xlabel("t-SNE Dimension 1"); plt.ylabel("t-SNE Dimension 2")
    plt.legend(title='Modality', fontsize=11, title_fontsize=12)
    savefig(fname)
safe_run("Figure: t-SNE before alignment", fig_tsne, emb_before, "Latent Space BEFORE Adversarial Alignment", "tsne_before")
safe_run("Figure: t-SNE after alignment", fig_tsne, emb_after, "Latent Space AFTER Adversarial Alignment (Unified Space)", "tsne_after")

def fig_confusion(y_true, y_pred, classes, title, fname):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    plt.figure(figsize=(7.5, 6.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 13, "weight": "bold"}, linewidths=0.8, linecolor='gray', cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=15, fontweight='bold'); plt.xlabel("Predicted Label"); plt.ylabel("True Label")
    plt.xticks(rotation=30, ha='right'); plt.yticks(rotation=0)
    savefig(fname)
safe_run("Figure: vision confusion matrix", fig_confusion, y_vis_te, vis_preds, vision_classes, "Vision Branch Confusion Matrix", "confusion_vision")
safe_run("Figure: EEG confusion matrix", fig_confusion, eeg_y_te, eeg_preds, eeg_classes, "EEG Branch Confusion Matrix", "confusion_eeg")
safe_run("Figure: clinical confusion matrix", fig_confusion, clin_y_te, clin_preds, clin_classes, "Clinical Decision Confusion Matrix", "confusion_clinical")

def fig_roc(y_true, probs, classes, title, fname):
    n_classes = len(classes)
    y_bin = np.eye(n_classes)[y_true]
    plt.figure(figsize=(7.5, 6.5))
    for i in range(n_classes):
        if len(np.unique(y_bin[:, i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        plt.plot(fpr, tpr, linewidth=2.5, label=f"{classes[i]} (AUC={auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5)
    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(fontsize=10, loc='lower right'); plt.grid(alpha=0.3)
    savefig(fname)
safe_run("Figure: vision ROC", fig_roc, y_vis_te, vis_probs, vision_classes, "Vision Branch ROC Curves", "roc_vision")
safe_run("Figure: EEG ROC", fig_roc, eeg_y_te, eeg_probs, eeg_classes, "EEG Branch ROC Curves", "roc_eeg")
safe_run("Figure: clinical ROC", fig_roc, clin_y_te, clin_probs, clin_classes, "Clinical Decision ROC Curves", "roc_clinical")

def fig_eeg_profiles():
    plt.figure(figsize=(11, 6))
    for c in range(N_EEG):
        mask = eeg_y == c
        profile = eeg_X_scaled_full[mask].mean(axis=0)[:min(20, eeg_X_scaled_full.shape[1])]
        plt.plot(profile, marker='o', markersize=5, linewidth=2.2, label=str(eeg_classes[c]))
    plt.title("Mean EEG Feature Profile per Cognitive State", fontsize=15, fontweight='bold')
    plt.xlabel("Feature Index"); plt.ylabel("Standardized Amplitude"); plt.legend(fontsize=11); plt.grid(alpha=0.3)
    savefig("eeg_feature_profiles")
safe_run("Figure: EEG feature profiles", fig_eeg_profiles)

def fig_clinical_corr():
    n = min(clin_X_scaled_full.shape[1], 15)
    df = pd.DataFrame(clin_X_scaled_full[:, :n], columns=[f"F{i}" for i in range(n)])
    plt.figure(figsize=(9, 7.5))
    sns.heatmap(df.corr(), cmap='coolwarm', center=0, annot=False, linewidths=0.3, cbar_kws={'label': 'Correlation'})
    plt.title("Clinical Feature Correlation Matrix", fontsize=15, fontweight='bold')
    savefig("clinical_correlation")
safe_run("Figure: clinical correlation heatmap", fig_clinical_corr)

def fig_accuracy_comparison():
    branches = ['Vision', 'EEG', 'Clinical']
    accs = [vis_acc, eeg_acc, clin_acc]
    plt.figure(figsize=(7.5, 6.5))
    bars = plt.bar(branches, accs, color=['#3B5998', '#2E8B57', '#B22222'], edgecolor='black', linewidth=1.3, width=0.55)
    for b, a in zip(bars, accs):
        plt.text(b.get_x() + b.get_width()/2, a + 0.015, f"{a:.3f}", ha='center', fontsize=13, fontweight='bold')
    plt.ylim(0, 1.08)
    plt.title("Final Test Accuracy per Modality Branch", fontsize=15, fontweight='bold')
    plt.ylabel("Accuracy"); plt.grid(axis='y', alpha=0.3)
    savefig("final_accuracy_comparison")
safe_run("Figure: accuracy comparison", fig_accuracy_comparison)

def fig_results_table():
    fig, ax = plt.subplots(figsize=(13, 2.2 + 0.5 * len(results_df)))
    ax.axis('off')
    tbl = ax.table(cellText=results_df.values, colLabels=results_df.columns,
                    cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 2.2)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('#888888')
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#3B5998')
        elif row == len(results_df):
            cell.set_facecolor('#FFF2CC')
            cell.set_text_props(weight='bold')
        else:
            cell.set_facecolor('#F5F5F5' if row % 2 == 0 else 'white')
    ax.set_title("Table 1. NeuroDecide-GAN Performance Summary Across Modality Branches",
                 fontsize=14, fontweight='bold', pad=20)
    savefig("results_table")
safe_run("Figure: results table", fig_results_table)

# ============================================================
# 8. ZIP + DOWNLOAD (figures + tables)
# ============================================================
def zip_and_download():
    zip_path = "/content/NeuroDecideGAN_outputs.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for folder in [FIG_DIR, TAB_DIR]:
            for root, _, files in os.walk(folder):
                for file in files:
                    fp = os.path.join(root, file)
                    arcname = os.path.join(os.path.basename(folder), os.path.relpath(fp, folder))
                    zf.write(fp, arcname)
    print(f"\nZipped {FIG_COUNT[0]} figures + results table -> {zip_path}")
    try:
        from google.colab import files as colab_files
        colab_files.download(zip_path)
    except Exception as e:
        print(f"[INFO] Auto-download unavailable ({e}). File is saved at: {zip_path}")

safe_run("Zipping and downloading outputs", zip_and_download)
print("\n=== NeuroDecide-GAN pipeline complete ===")