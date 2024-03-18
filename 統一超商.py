import streamlit as st
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import os

class CSVPricePredictor:
    def __init__(self):
        self.model = None

    def train_model(self):
        # 使用本地 CSV 檔案訓練模型
        path2='/Users/user/Desktop/leetcode/統一超商/retail_price.csv'
        df=pd.read_csv(path2)

        # 生必要的 column 供後續訓練使用
        df['comp1_diff'] = df['unit_price'] - df['comp_1']
        df['comp2_diff'] = df['unit_price'] - df['comp_2']
        df['comp3_diff'] = df['unit_price'] - df['comp_3']
        df['fp1_diff'] = df['freight_price'] - df['fp1']
        df['fp2_diff'] = df['freight_price'] - df['fp2']
        df['fp3_diff'] = df['freight_price'] - df['fp3']

        # 資料前處理, 分類後捏特徵再合併
        cols_to_mean = ['product_id', 'month_year', 'comp1_diff', 'comp2_diff', 'comp3_diff',
                'fp1_diff', 'fp2_diff', 'fp3_diff', 'product_score', 'unit_price']
        cols_to_sum = ['product_id', 'total_price', 'freight_price', 'customers']
        mean_df = df[cols_to_mean]
        sum_df = df[cols_to_sum]

        products_mean = mean_df.groupby('product_id').mean()
        products_sum = sum_df.groupby('product_id').sum()
        products = pd.concat([products_mean, products_sum], axis=1).reset_index()
        X, y = products.drop(['product_id', 'unit_price'], axis=1), products['unit_price']

        # 創建模型
        lasso = Lasso(alpha=0.1)
        lasso.fit(X, y)
        rf = RandomForestRegressor(n_estimators=50, random_state=19)
        rf.fit(X, y)
        self.model = (lasso, rf)

    def predict(self, df):
        
        # 預測定價
        lasso_pred = self.model[0].predict(df)
        rf_pred = self.model[1].predict(df)
        final_pred = (lasso_pred + rf_pred) / 2
        final_pred_scalar = round(abs(final_pred[0])) 

        return final_pred_scalar
    
    def AI_agent(self,upload_csv,pred_price):
        #設定環境變數供langchain使用
        path = '/Users/user/Desktop/leetcode/統一超商/Gemini_API.txt'
        with open(path,'r') as f:
            api_key=f.read().strip()
        os.environ['GOOGLE_API_KEY'] = api_key

        #方法2 systemmessage設定角色，humanmessage提問, 設定cache取得上一輪對話內容
        model = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
        message=model(
            [
                SystemMessage(content=f"你是data scientist,很擅長分析資料跟給出business insight,以下資料是商品的資訊{upload_csv}"),
                HumanMessage(content=f"嘗試說明為何這個商品的定價要為{pred_price}最適合"),
            ]
        )
        return(message.content)

def main():
    st.set_page_config(page_title="請上傳CSV")
    st.header("請上傳CSV")

    user_csv = st.file_uploader("上傳檔案，系統會自動定價", type="csv")

    if user_csv is not None:
        # 讀取上傳的 CSV 檔案
        df_upload = pd.read_csv(user_csv)

        #新增特徵給df_upload
        df_upload['comp1_diff'] = df_upload['unit_price'] - df_upload['comp_1']
        df_upload['comp2_diff'] = df_upload['unit_price'] - df_upload['comp_2']
        df_upload['comp3_diff'] = df_upload['unit_price'] - df_upload['comp_3']
        df_upload['fp1_diff'] = df_upload['freight_price'] - df_upload['fp1']
        df_upload['fp2_diff'] = df_upload['freight_price'] - df_upload['fp2']
        df_upload['fp3_diff'] = df_upload['freight_price'] - df_upload['fp3']

        # 資料前處理, 分類後捏特徵再合併
        cols_to_mean = ['product_id', 'month_year', 'comp1_diff', 'comp2_diff', 'comp3_diff',
                'fp1_diff', 'fp2_diff', 'fp3_diff', 'product_score', 'unit_price']
        cols_to_sum = ['product_id', 'total_price', 'freight_price', 'customers']
        mean_df = df_upload[cols_to_mean]
        sum_df = df_upload[cols_to_sum]

        products_mean = mean_df.groupby('product_id').mean()
        products_sum = sum_df.groupby('product_id').sum()
        products_upload = pd.concat([products_mean, products_sum], axis=1).reset_index()       
        X= products_upload.drop(['product_id', 'unit_price'], axis=1)

        # 初始化並訓練模型
        predictor = CSVPricePredictor()
        predictor.train_model()

        # 預測定價
        final_pred = predictor.predict(X)

        #呼叫AI解釋定價
        AIresponse=CSVPricePredictor()
        Explaination=AIresponse.AI_agent(X,final_pred)

        # 輸出預測結果
        st.write(f"此商品建議售價:{final_pred}")
        st.write(f"原因是:{Explaination}")

if __name__ == "__main__":
    main()
