import streamlit as st
import pandas as pd
import pickle

###### Giao diện Streamlit ######
st.image('w05227-small.jpg', use_container_width =True)

# function cần thiết
def lookup_member(df, member_number):
    result = df[df['Member_number'].astype(str) == str(member_number)]
    if result.empty:
        return f"Không tìm thấy thành viên với Member_number = {member_number}"
    return result

def assign_cluster_names(df):
    cluster_name_map = {
        0: "Khách hàng trung bình",
        1: "Khách hàng không hoạt động",
        2: "Khách hàng VIP",
        3: "Khách hàng trung bình-thấp",
        4: "Khách hàng không hoạt động"
    }
    df['cluster_names'] = df['Cluster'].map(cluster_name_map)
    return df

def get_recommendations(df, ma_san_pham, cosine_sim, nums=5):
    matching_indices = df.index[df['ma_san_pham'] == ma_san_pham].tolist()
    if not matching_indices:
        print(f"No product found with ID: {ma_san_pham}")
        return pd.DataFrame()
    idx = matching_indices[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:nums+1]
    product_indices = [i[0] for i in sim_scores]
    return df.iloc[product_indices]

# Đọc dữ liệu giao dịch
df_trans = pd.read_csv('data_segmentation_total.csv')
df_trans[df_trans.duplicated(keep=False)]

# Lấy 10 sản phẩm
random_trans = df_trans.head(n=15)
st.session_state.random_trans = random_trans

# Feature Engineering
df_trans['Gross'] = df_trans.price * df_trans['items']
df_trans['Order_id'] = df_trans.index
df_trans['Date'] = pd.to_datetime(df_trans['Date'])

# RFM
Recency = lambda x : (df_trans['Date'].max().date() - x.max().date()).days
Frequency  = lambda x: len(x.unique())
Monetary = lambda x : round(sum(x), 2)

data_RFM = df_trans.groupby('Member_number').agg({'Date': Recency,
                                                 'Order_id': Frequency,
                                                 'Gross': Monetary })
data_RFM.columns = ['Recency', 'Frequency', 'Monetary']
data_RFM = data_RFM.sort_values('Monetary', ascending=False)

# Load K-Means model
with open('model_Kmeans_seg.pkl', 'rb') as f:
    model_Kmeans_seg = pickle.load(f)

df_now = data_RFM[['Recency','Frequency','Monetary']].copy()
model_predict = model_Kmeans_seg.predict(df_now)
df_now["Cluster"] = model_predict
df_now = assign_cluster_names(df_now)
df_now = df_now.reset_index()


# Kiểm tra xem 'selected_ma_san_pham' đã có trong session_state hay chưa
if 'selected_ma_san_pham' not in st.session_state:
    st.session_state.selected_ma_san_pham = None

# Dropdown chọn mã khách hàng
memb_options = [(row['Member_number'], row['Category']) for index, row in st.session_state.random_trans.iterrows()]
st.session_state.random_trans
selected_member_tuple = st.selectbox(
    "Chọn mã khách hàng",
    options=memb_options,
    format_func=lambda x: x[0]  # Hiển thị mã khách hàng
)
st.write("Bạn đã chọn:", selected_member_tuple)

if selected_member_tuple:
    selected_member_number = selected_member_tuple[0] # Lấy Member_number từ tuple
    st.write("Member: ", selected_member_number)

    # Tìm thông tin phân hạng khách hàng
    segment_info = lookup_member(df_now, selected_member_number)

    st.write('##### Phân hạng khách hàng:')
    if isinstance(segment_info, pd.DataFrame):
        st.write(segment_info['cluster_names'].values[0]) # Truy cập giá trị đầu tiên của cột
    else:
        st.write(segment_info) # Hiển thị thông báo nếu không tìm thấy
else:
    st.write("Vui lòng chọn một mã khách hàng.")

# (Phần code gợi ý sản phẩm giữ nguyên vì nó không liên quan đến lỗi này)