import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set()
    sns_available = True
except Exception:
    sns_available = False

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             mean_absolute_error, mean_squared_error, r2_score)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier, AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# CONFIG
DATA_PATH = r"D:\STians_final_with_updated_tutors_modified_final (1).xlsx"
TARGET_COLUMN = None
FORCE_PROBLEM = "regression"
SAVE_PLOTS = False
RANDOM_STATE = 42
SAMPLE_PLOT_N = 1000


def ensure_plot_dir():
    if SAVE_PLOTS:
        os.makedirs("plots", exist_ok=True)


def save_or_show(fig, name):
    if SAVE_PLOTS:
        path = os.path.join("plots", name + ".png")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved plot: {path}")
    else:
        plt.show()


def load_data(path):
    print("Loading:", path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = path.split('.')[-1].lower()
    if ext in ['xls', 'xlsx']:
        df = pd.read_excel(path)
    elif ext == 'csv':
        df = pd.read_csv(path)
    else:
        try:
            df = pd.read_excel(path)
        except Exception:
            df = pd.read_csv(path)
    print("Data shape:", df.shape)
    return df


def kpi_cards(df):
    total_records = df.shape[0]
    total_features = df.shape[1]
    numerical_features = df.select_dtypes(include=[np.number]).shape[1]
    categorical_features = df.select_dtypes(include=['object', 'category']).shape[1]
    missing = df.isnull().sum().sum()
    total_cells = df.size
    missing_pct = (missing / total_cells) * 100 if total_cells else 0
    print(f"Records: {total_records} | Features: {total_features} | Numeric: {numerical_features} | Categorical: {categorical_features} | Missing%: {missing_pct:.2f}")


def data_understanding(df, n=5):
    print('\n-- head --')
    print(df.head(n))
    print('\n-- info --')
    df.info()
    print('\n-- describe --')
    print(df.describe(include='all').transpose())


def missing_value_analysis(df):
    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        print('\nNo missing values')
    else:
        print('\nMissing by column:')
        print(miss)
        if sns_available:
            plt.figure(figsize=(8,4))
            sns.heatmap(df[miss.index].isnull(), cbar=False)
            plt.title('Missing values')
            save_or_show(plt.gcf(), 'missing_heatmap')


def data_analysis(df):
    # consider only numeric columns that are likely features (exclude id-like columns)
    id_like = [c for c in df.columns if any(tok in c.lower() for tok in ['id','regi','registration','name','father','srno','serial'])]
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in id_like]
    if not numeric_cols:
        print(" -> No numeric columns for skewness/outlier analysis.")
        return

    # define academic columns heuristically: marks, attendance, homework, tests, study_hours, fees
    academic_tokens = ['mark','attendance','homework','test','study','fee','fees','score','percentage','percent']
    academic_cols = [c for c in numeric_cols if any(t in c.lower() for t in academic_tokens)]
    if not academic_cols:
        # fallback to numeric cols without id-like
        academic_cols = numeric_cols

    print("\n-- Skewness (selected numeric columns) --")
    print(df[academic_cols].skew().sort_values(ascending=False))

    # boxplots for academic columns only
    sample = df[academic_cols].dropna()
    if sample.shape[0] > SAMPLE_PLOT_N:
        sample = sample.sample(SAMPLE_PLOT_N, random_state=RANDOM_STATE)

    plt.figure(figsize=(12,6))
    if sns_available:
        sns.boxplot(data=sample, orient='h')
    else:
        sample.plot(kind='box', vert=False, figsize=(12,6))
    plt.title("Boxplots for academic numeric features (outlier check)")
    save_or_show(plt.gcf(), "academic_boxplots")



def visualizations(df):
    # exclude id-like columns before selecting numerics / categories
    id_like = [c for c in df.columns if any(tok in c.lower() for tok in ['id','regi','registration','name','father','srno','serial'])]
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in id_like]
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # 1) Histogram of Class (keep)
    if 'Class' in df.columns:
        plt.figure(figsize=(8,4))
        df['Class'].dropna().astype(int).value_counts().sort_index().plot(kind='bar')
        plt.title("Histogram of Class")
        save_or_show(plt.gcf(), "hist_class")

    # 2) Correlation heatmap on numeric_cols but drop registration-like numeric columns
    if len(numeric_cols) >= 2:
        corr_cols = numeric_cols.copy()
        # further remove obvious identifiers like Registration No or Student ID if present
        corr_cols = [c for c in corr_cols if not any(tok in c.lower() for tok in ['registration','student id','student_id','reg no'])]
        if len(corr_cols) >= 2:
            corr = df[corr_cols].corr()
            plt.figure(figsize=(10,8))
            if sns_available:
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
            else:
                plt.imshow(corr, cmap='coolwarm', aspect='auto'); plt.colorbar()
            plt.title("Correlation heatmap (academic features)")
            save_or_show(plt.gcf(), "correlation_heatmap")

    # 3) Scatter examples: choose meaningful pairs if present
    pairs = [('Study_Hours','Maths Marks'), ('Study_Hours','Science Marks'), ('Attendance %','Class Tests Attended')]
    for xcol,ycol in pairs:
        if xcol in df.columns and ycol in df.columns:
            plt.figure(figsize=(7,5))
            plt.scatter(df[xcol], df[ycol], alpha=0.6)
            plt.xlabel(xcol); plt.ylabel(ycol)
            plt.title(f"Scatter: {xcol} vs {ycol}")
            save_or_show(plt.gcf(), f"scatter_{xcol}_{ycol}")

    # 4) Categorical summary: choose first categorical column with reasonably small cardinality
    chosen_cat = None
    for c in cat_cols:
        if df[c].nunique() <= 25 and df[c].nunique() > 1 and 'name' not in c.lower():
            chosen_cat = c
            break
    if chosen_cat:
        vc = df[chosen_cat].value_counts().head(15)
        plt.figure(figsize=(8,4))
        vc.plot(kind='bar')
        plt.title(f"Counts of {chosen_cat} (top categories)")
        save_or_show(plt.gcf(), "cat_counts")

        
def feature_engineering(df, drop_id_like=True, onehot_threshold=10, scale_method='standard'):
    data = df.copy()
    data.columns = [c.strip() for c in data.columns]
    if drop_id_like:
        id_like = [c for c in data.columns if c.lower() in ['id', 'srno', 'serial', 's.no', 's_no', 'index']]
        if id_like:
            data = data.drop(columns=id_like)
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    for c in cat_cols:
        if data[c].nunique() <= 2:
            le = LabelEncoder()
            data[c] = le.fit_transform(data[c].astype(str))
        elif data[c].nunique() <= onehot_threshold:
            d = pd.get_dummies(data[c].astype(str), prefix=c, drop_first=True)
            data = pd.concat([data.drop(columns=[c]), d], axis=1)
        else:
            le = LabelEncoder()
            data[c] = le.fit_transform(data[c].astype(str))
    num_cols_after = data.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols_after:
        scaler = StandardScaler() if scale_method == 'standard' else MinMaxScaler()
        data[num_cols_after] = scaler.fit_transform(data[num_cols_after])
    else:
        scaler = None
    return data, scaler


def detect_target_and_problem(df):
    cols = df.columns.tolist()
    if TARGET_COLUMN and TARGET_COLUMN in cols:
        target = TARGET_COLUMN
    else:
        common_names = ['target', 'label', 'y', 'outcome', 'class', 'result', 'status']
        found = [c for c in cols if c.lower() in common_names]
        if found:
            target = found[0]
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            target = None
            for c in reversed(numeric_cols):
                if df[c].nunique() >= 2:
                    target = c
                    break
    if target is None:
        return None, 'unsupervised'
    if FORCE_PROBLEM:
        return target, FORCE_PROBLEM
    if pd.api.types.is_numeric_dtype(df[target]):
        nunique = df[target].nunique(dropna=True)
        problem = "regression"

    else:
        problem = 'classification'
    print(f"Target: {target} | Problem: {problem}")
    return target, problem


def prepare_supervised(df, target):
    data = df.copy()

    # remove identifier columns (not useful for learning)
    id_like = [c for c in data.columns if any(
        x in c.lower() for x in ['id', 'regi', 'registration', 'name', 'father', 'srno', 'serial']
    )]
    data = data.drop(columns=id_like, errors='ignore')

    data = data.dropna(subset=[target])



    
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    if num_cols:
        imputer = SimpleImputer(strategy='mean')
        data[num_cols] = imputer.fit_transform(data[num_cols])
    if cat_cols:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        data[cat_cols] = imputer_cat.fit_transform(data[cat_cols])
    for c in cat_cols:
        le = LabelEncoder()
        data[c] = le.fit_transform(data[c].astype(str))
    X = data.drop(columns=[target])
    y = data[target]
    print('X,y shapes:', X.shape, y.shape)
    return X, y


def run_classification_models(X, y):
    strat = y if len(np.unique(y)) > 1 and len(np.unique(y)) < len(y) else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=strat)
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    if num_cols:
        X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'NaiveBayes': GaussianNB(),
        'DecisionTree': DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        'SVM': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    }
    results = {}
    for name, model in models.items():
        print('\nTraining', name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f'Accuracy {name}: {acc:.4f}')
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, zero_division=0))
        try:
            cv = cross_val_score(model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), scoring='accuracy')
            print(f'5-fold CV mean={cv.mean():.4f} std={cv.std():.4f}')
        except Exception:
            pass
        results[name] = acc
    try:
        bag = BaggingClassifier(n_estimators=30, random_state=RANDOM_STATE)
        bag.fit(X_train, y_train)
        print('Bagging acc:', accuracy_score(y_test, bag.predict(X_test)))
        ada = AdaBoostClassifier(n_estimators=50, random_state=RANDOM_STATE)
        ada.fit(X_train, y_train)
        print('AdaBoost acc:', accuracy_score(y_test, ada.predict(X_test)))
    except Exception:
        pass
    return results


def run_regression_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    if num_cols:
        X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])
    from sklearn.linear_model import LinearRegression
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    y_pred = lin.predict(X_test)
    print('Linear MAE:', mean_absolute_error(y_test, y_pred), 'R2:', r2_score(y_test, y_pred))
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2)
    Xp_train = poly.fit_transform(X_train)
    Xp_test = poly.transform(X_test)
    lin2 = LinearRegression()
    lin2.fit(Xp_train, y_train)
    y2 = lin2.predict(Xp_test)
    print('Poly deg2 MAE:', mean_absolute_error(y_test, y2), 'R2:', r2_score(y_test, y2))
    rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    yrf = rf.predict(X_test)
    print('RF R2:', r2_score(y_test, yrf))
    return {'Linear': lin, 'Poly': lin2, 'RF': rf}


def unsupervised_analysis(df):
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num:
        print('No numeric cols for unsupervised')
        return
    X = df[num].dropna()
    if X.shape[0] > SAMPLE_PLOT_N:
        Xs = X.sample(SAMPLE_PLOT_N, random_state=RANDOM_STATE)
    else:
        Xs = X
    scaler = StandardScaler()
    Xs_s = scaler.fit_transform(Xs)
    sse = []
    for k in range(1,7):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        kmeans.fit(Xs_s)
        sse.append(kmeans.inertia_)
    plt.figure(figsize=(6,4))
    plt.plot(range(1,7), sse, marker='o')
    plt.title('Elbow')
    save_or_show(plt.gcf(), 'elbow')
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(Xs_s)
    plt.figure(figsize=(7,5))
    plt.scatter(Xs_s[:,0], Xs_s[:,1], c=labels, alpha=0.6)
    plt.title('KMeans k=3')
    save_or_show(plt.gcf(), 'kmeans')
    pca = PCA(n_components=min(6, Xs_s.shape[1]))
    p = pca.fit_transform(Xs_s)
    print('PCA explained variance:', pca.explained_variance_ratio_)
    plt.figure(figsize=(6,4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.title('PCA cumulative')
    save_or_show(plt.gcf(), 'pca')


def final_report_summary():
    print('Review metrics above and choose best model.')


def main():
    ensure_plot_dir()
    print('INT234: Predictive Analytics - Dashboard')
    df = load_data(DATA_PATH)
    kpi_cards(df)
    data_understanding(df)
    missing_value_analysis(df)
    data_analysis(df)
    visualizations(df)
    df_fe, _ = feature_engineering(df, drop_id_like=True)
    target, problem = detect_target_and_problem(df)
    if problem == 'unsupervised':
        unsupervised_analysis(df_fe)
        final_report_summary()
        return
    X, y = prepare_supervised(df, target)
    Xy, _ = feature_engineering(pd.concat([X, y], axis=1), drop_id_like=False)
    if target in Xy.columns:
        y_final = Xy[target]
        X_final = Xy.drop(columns=[target])
    else:
        X_final, y_final = X, y
    if problem == 'classification':
        run_classification_models(X_final, y_final)
    else:
        run_regression_models(X_final, y_final)
    unsupervised_analysis(df_fe)
    final_report_summary()


if __name__ == '__main__':
    main()
