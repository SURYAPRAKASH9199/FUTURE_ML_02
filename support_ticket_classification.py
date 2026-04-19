# ============================================================
# SUPPORT TICKET CLASSIFICATION & PRIORITIZATION SYSTEM
# Future Interns - ML Task 02
# ============================================================
import pandas as pd, numpy as np, matplotlib, re, string, os, warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import joblib

PALETTE = ["#6C5CE7","#00B894","#FDCB6E","#E17055","#74B9FF","#FD79A8"]
STOP_WORDS = set(ENGLISH_STOP_WORDS)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+','',text)
    text = re.sub(r'\d+','',text)
    text = text.translate(str.maketrans('','',string.punctuation))
    return ' '.join(t for t in text.split() if t not in STOP_WORDS and len(t)>2)

# ── DATASET ──────────────────────────────────────────────────
print("Loading dataset...")
if os.path.exists("customer_support_tickets.csv"):
    df = pd.read_csv("customer_support_tickets.csv")
    print(f"Dataset loaded: {df.shape}")
else:
    print("Kaggle CSV not found - using synthetic dataset\n")
    np.random.seed(42)
    cats = {
        "Billing": [
            "I was charged twice for my subscription this month please help",
            "My invoice shows wrong amount please fix it urgently",
            "I need a refund for the unauthorized charge on my account",
            "Payment failed but money was deducted from my account",
            "Where is my billing statement for last month I need it",
            "My credit card was charged without my authorization ASAP",
            "Can I get a discount on my annual subscription plan",
            "I cancelled my subscription but still got charged refund",
            "My promo code did not apply to my bill please check",
            "Invoice amount does not match what I agreed to pay",
        ],
        "Technical Issue": [
            "The application crashes every time I try to upload files",
            "I cannot login and my password reset email not working",
            "The dashboard is loading very slowly today not working",
            "Error 500 appears every time I try to submit the form",
            "My data is not syncing between my devices issue",
            "The mobile app keeps freezing on the home screen crash",
            "API integration is returning wrong response error codes",
            "I cannot export data the export button does not work",
            "Two factor authentication is not sending SMS not working",
            "Database connection timeout error showing in my logs",
        ],
        "Account": [
            "I want to change my email address on my account please",
            "How do I add a new team member to my workspace today",
            "I need to delete my account and remove all my data",
            "My account got locked after too many login attempts",
            "Please update my company name in my account profile",
            "I want to merge two of my accounts together into one",
            "How do I enable single sign on SSO for my team",
            "I cannot access my previous account transaction history",
            "How do I revoke access for a former team member account",
            "I forgot my username and I cannot recover my account",
        ],
        "General Query": [
            "What features are included in the premium subscription plan",
            "How long does it normally take to process a refund request",
            "Do you offer a free trial period for new users signup",
            "What are your customer support team working hours",
            "Can I use your service in multiple countries worldwide",
            "Is there a mobile application available for download",
            "How do I integrate your product with third party tools",
            "What is your data privacy and security policy details",
            "Are there video tutorials available for beginner users",
            "How many users can I add to my current subscription plan",
        ],
    }
    HIGH_KW = ["urgent","asap","crash","error","failed","not working","cannot login","charged twice","unauthorized","500","freeze","timeout"]
    MED_KW  = ["slow","wrong","update","change","issue","problem","delay","upgrade","missing","not syncing","not sending"]
    def ap(t):
        t=t.lower()
        if any(k in t for k in HIGH_KW): return "High"
        if any(k in t for k in MED_KW):  return "Medium"
        return "Low"
    rows=[]
    for cat,texts in cats.items():
        for i in range(150):
            t = texts[i % len(texts)]
            rows.append({"ticket_text": t, "category": cat, "priority": ap(t)})
    df = pd.DataFrame(rows).sample(frac=1,random_state=42).reset_index(drop=True)
    df["ticket_id"] = [f"TKT-{1000+i}" for i in range(len(df))]
    print(f"Synthetic dataset: {df.shape}")

text_col = next((c for c in df.columns if any(k in c.lower() for k in ["text","description","body","subject"])), df.columns[0])
cat_col  = next((c for c in df.columns if any(k in c.lower() for k in ["category","type","topic","label"])), None)
pri_col  = "priority" if "priority" in df.columns else None

if pri_col is None:
    HIGH_KW=["urgent","asap","crash","error","failed","not working","unauthorized"]
    MED_KW=["slow","wrong","update","issue","problem","delay"]
    df["priority"]=df[text_col].apply(lambda t:"High" if any(k in str(t).lower() for k in HIGH_KW) else ("Medium" if any(k in str(t).lower() for k in MED_KW) else "Low"))
    pri_col="priority"

print(f"text='{text_col}'  category='{cat_col}'  priority='{pri_col}'")
print(f"Shape: {df.shape}\n{df[cat_col].value_counts()}\n{df[pri_col].value_counts()}")

print("\nCleaning text...")
df["cleaned"] = df[text_col].apply(clean_text)

# ── PLOTS ────────────────────────────────────────────────────
os.makedirs("plots",exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

fig,axes=plt.subplots(1,2,figsize=(14,5))
fig.suptitle("Support Ticket Dataset Overview",fontsize=16,fontweight="bold")
cc=df[cat_col].value_counts()
bars=axes[0].bar(cc.index,cc.values,color=PALETTE[:len(cc)],edgecolor="white")
axes[0].set_title("Tickets per Category",fontsize=13,fontweight="bold")
axes[0].tick_params(axis="x",rotation=20)
for bar,v in zip(bars,cc.values): axes[0].text(bar.get_x()+bar.get_width()/2,v+1,str(v),ha="center",fontweight="bold")
pc=df[pri_col].value_counts()
pcols=[{"High":"#E17055","Medium":"#FDCB6E","Low":"#00B894"}.get(p,"#74B9FF") for p in pc.index]
axes[1].pie(pc.values,labels=pc.index,colors=pcols,autopct="%1.1f%%",startangle=140,wedgeprops=dict(edgecolor="white",linewidth=2))
axes[1].set_title("Priority Distribution",fontsize=13,fontweight="bold")
plt.tight_layout(); plt.savefig("plots/01_distribution.png",dpi=150,bbox_inches="tight"); plt.close()
print("Saved plots/01_distribution.png")

fig,ax=plt.subplots(figsize=(10,5))
sns.heatmap(pd.crosstab(df[cat_col],df[pri_col]),annot=True,fmt="d",cmap="YlOrRd",ax=ax,linewidths=0.5,linecolor="white")
ax.set_title("Category x Priority Heatmap",fontsize=14,fontweight="bold")
plt.tight_layout(); plt.savefig("plots/02_heatmap.png",dpi=150,bbox_inches="tight"); plt.close()
print("Saved plots/02_heatmap.png")

# ── ENCODE & SPLIT ───────────────────────────────────────────
le_cat=LabelEncoder(); le_pri=LabelEncoder()
df["cat_enc"]=le_cat.fit_transform(df[cat_col])
df["pri_enc"]=le_pri.fit_transform(df[pri_col])

X_tr,X_te,yc_tr,yc_te,yp_tr,yp_te=train_test_split(df["cleaned"],df["cat_enc"],df["pri_enc"],test_size=0.2,random_state=42,stratify=df["cat_enc"])
tfidf=TfidfVectorizer(max_features=5000,ngram_range=(1,2),min_df=1)
Xtr_v=tfidf.fit_transform(X_tr); Xte_v=tfidf.transform(X_te)
print(f"\nTF-IDF: train {Xtr_v.shape}  test {Xte_v.shape}")

MODELS={
    "Logistic Regression": LogisticRegression(max_iter=1000,C=1.0,random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200,random_state=42),
    "LinearSVC":           LinearSVC(max_iter=2000,random_state=42),
    "Naive Bayes":         MultinomialNB(alpha=0.5),
}

def train_all(y_tr,y_te,le,label):
    res={}
    print(f"\n{'='*50}\n  {label}\n{'='*50}")
    for name,clf in MODELS.items():
        clf.fit(Xtr_v,y_tr)
        preds=clf.predict(Xte_v)
        acc=accuracy_score(y_te,preds)
        present_labels=sorted(set(y_te)|set(preds))
        present_names=[le.classes_[i] for i in present_labels]
        res[name]=(acc,clf,preds)
        print(f"\n  {name}: {acc*100:.2f}%")
        print(classification_report(y_te,preds,labels=present_labels,target_names=present_names,zero_division=0))
    return res

cat_res=train_all(yc_tr,yc_te,le_cat,"CATEGORY CLASSIFICATION")
pri_res=train_all(yp_tr,yp_te,le_pri,"PRIORITY PREDICTION")

best_cat=max(cat_res,key=lambda n:cat_res[n][0]); best_pri=max(pri_res,key=lambda n:pri_res[n][0])
print(f"\nBest Category: {best_cat} ({cat_res[best_cat][0]*100:.2f}%)")
print(f"Best Priority: {best_pri} ({pri_res[best_pri][0]*100:.2f}%)")

names=list(MODELS.keys())
fig,axes=plt.subplots(1,2,figsize=(14,5))
fig.suptitle("Model Accuracy Comparison",fontsize=15,fontweight="bold")
for ax,accs,title in zip(axes,[[cat_res[n][0]*100 for n in names],[pri_res[n][0]*100 for n in names]],["Category","Priority"]):
    bars=ax.barh(names,accs,color=PALETTE[:len(names)],edgecolor="white")
    ax.set_title(title,fontweight="bold"); ax.set_xlim(0,115)
    for bar,v in zip(bars,accs): ax.text(v+0.5,bar.get_y()+bar.get_height()/2,f"{v:.1f}%",va="center",fontweight="bold")
plt.tight_layout(); plt.savefig("plots/03_model_comparison.png",dpi=150,bbox_inches="tight"); plt.close()
print("Saved plots/03_model_comparison.png")

fig,axes=plt.subplots(1,2,figsize=(14,6))
for ax,clf,y_te,le,title,cmap in [
    (axes[0],cat_res[best_cat][1],yc_te,le_cat,f"Category - {best_cat}","Blues"),
    (axes[1],pri_res[best_pri][1],yp_te,le_pri,f"Priority - {best_pri}","Oranges")]:
    cm=confusion_matrix(y_te,clf.predict(Xte_v),labels=sorted(set(y_te)))
    ConfusionMatrixDisplay(cm,display_labels=[le.classes_[i] for i in sorted(set(y_te))]).plot(ax=ax,colorbar=False,cmap=cmap)
    ax.set_title(title,fontweight="bold"); ax.tick_params(axis="x",rotation=25)
fig.suptitle("Confusion Matrices",fontsize=15,fontweight="bold")
plt.tight_layout(); plt.savefig("plots/04_confusion_matrices.png",dpi=150,bbox_inches="tight"); plt.close()
print("Saved plots/04_confusion_matrices.png")

os.makedirs("models",exist_ok=True)
joblib.dump(cat_res[best_cat][1],"models/category_model.pkl")
joblib.dump(pri_res[best_pri][1],"models/priority_model.pkl")
joblib.dump(tfidf,"models/tfidf_vectorizer.pkl")
joblib.dump(le_cat,"models/label_encoder_category.pkl")
joblib.dump(le_pri,"models/label_encoder_priority.pkl")
print("Models saved to /models/")

CE={"Billing":"💳","Technical Issue":"🔧","Account":"👤","General Query":"❓"}
PE={"High":"🔴","Medium":"🟡","Low":"🟢"}

def predict_ticket(txt):
    v=tfidf.transform([clean_text(txt)])
    cat=le_cat.inverse_transform(cat_res[best_cat][1].predict(v))[0]
    pri=le_pri.inverse_transform(pri_res[best_pri][1].predict(v))[0]
    return {"ticket":txt,"category":f"{CE.get(cat,'📋')} {cat}","priority":f"{PE.get(pri,'⚪')} {pri}"}

demos=["My account was charged twice ASAP I need a refund!",
       "How do I reset my password I forgot it completely.",
       "The app crashes every time I open my reports error.",
       "What features are in the pro plan pricing details?",
       "I need to add a new team member to our workspace.",
       "Payment failed but money was deducted from my bank.",
       "Error 500 appears when I try to submit contact form.",
       "Do you offer a free trial period for new users?"]

print("\n"+"="*55+"\nLIVE TICKET PREDICTIONS\n"+"="*55)
for t in demos:
    r=predict_ticket(t)
    print(f"\n  Ticket  : {r['ticket']}")
    print(f"  Category: {r['category']}")
    print(f"  Priority: {r['priority']}")

print(f"\n{'='*55}\nFINAL SUMMARY\n{'='*55}")
print(f"  Total tickets  : {len(df)}")
print(f"  Categories     : {list(df[cat_col].unique())}")
print(f"  Priority levels: {sorted(df[pri_col].unique())}")
print(f"\n  Best Category  : {best_cat}  ({cat_res[best_cat][0]*100:.2f}%)")
print(f"  Best Priority  : {best_pri}  ({pri_res[best_pri][0]*100:.2f}%)")
print("\nProject complete! Ready for GitHub!")
print("="*55)
