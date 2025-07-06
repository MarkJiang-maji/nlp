 LLM AI-Generated Text Detection 
本專案參加的是 Kaggle 競賽：Detect AI-generated Text，目標是分辨「人類寫的文章」與「大型語言模型（LLM）生成的文章」。
本專案主要參考並改進自 Kaggle 上的解法：Bidirectional LSTM + Transformer CNN Approach。
主要重點
✨ 使用 Bidirectional LSTM 搭配簡易的 Transformer Block 與 CNN


💬 同時結合 文字序列輸入 及 手工統計特徵（如大寫比例、標點符號比例、字數）


📊 包含資料探索、視覺化（如長度分佈、TF-IDF、情感分析、WordCloud）


✅ 可直接在 Colab 執行，只需登入 Kaggle 帳號即可自動下載所有資料集



專案內容說明
本專案主要分成幾個部分：
1. 資料下載與處理
透過 kagglehub 自動下載競賽主資料集與 10 多個外部資料集，整合並平衡人類與 AI 寫作樣本，總數超過 6 萬筆以上。
2. EDA（探索性資料分析）
文字長度分布圖（Histogram & Boxplot）


最常見單字（含排除 Stopwords 的版本）


WordCloud 視覺化


標點符號頻率分析


情感分數分析（使用 VADER）


3. 特徵工程
使用 TextVectorization + 自定標準化規則（分詞+去標點）


加入統計特徵（字數、標點比例、大寫比例）做為第二輸入


4. 模型架構
模型採用雙輸入架構：
Input 1：LSTM 處理文字序列 + GlobalMaxPooling


Input 2：三個手工特徵


中間合併後經過全連接層 + Dropout，最後輸出 Binary 分類結果



所有程式都可以在kaggle直接執行，如果要用colab跑請照以下步驟執行: 
你可以直接在 Google Colab 上執行本程式碼，只需以下步驟：
✅ 步驟 1：準備 Kaggle API 金鑰
登入 Kaggle，取得 username 與 key


✅ 步驟 2：在 Colab 中填入帳號
打開程式碼後，將以下內容中的帳號資訊填入：
os.environ['KAGGLE_USERNAME'] = "你的Kaggle帳號"
os.environ['KAGGLE_KEY'] = "你的Kaggle金鑰"

即可開始自動下載所有資料，進行訓練與預測。

📝 輸出結果
訓練完成後會對 test_essays.csv 做預測，產出 submission.csv 檔案如下：
id,generated
1,0.987
2,0.021
...


🔗 資源與參考
Kaggle 競賽頁面


參考原始模型（Bidirectional LSTM + Transformer CNN）
參考TF-IDF等作法 (7️⃣ SE7EN PROMPTS ✋✌️)



