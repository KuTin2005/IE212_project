import streamlit as st
import keras
import numpy as np
from PIL import Image

# CẤU HÌNH TRANG
st.set_page_config(page_title="Hệ thống Chẩn đoán Bệnh Lá Cây", layout="centered")
st.title("Trợ lý AI Chẩn đoán Bệnh Thực Vật")
st.write("Cơ chế **Ensemble Learning (Weighted Soft Voting)**")

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]
REMEDIES = {
    'Apple - Apple scab': "Biện pháp canh tác: Chủ động cắt tỉa cành để tăng độ thông thoáng, thu gom và tiêu hủy triệt để lá rụng cuối vụ nhằm cắt đứt nguồn nấm bệnh. Biện pháp hóa học: Phun phòng ngừa bằng các loại thuốc trừ nấm gốc đồng hoặc Mancozeb ngay khi chồi mới nhú.",
    'Apple - Black rot': "Biện pháp canh tác: Loại bỏ và tiêu hủy ngay lập tức các quả bị thối, cành khô hoặc vỏ cây bị nứt nẻ hoại tử. Biện pháp hóa học: Phun thuốc trừ nấm phổ rộng như Captan vào giai đoạn ngay sau khi rụng hoa, lặp lại chu kỳ 10-14 ngày nếu thời tiết ẩm ướt.",
    'Apple - Cedar apple rust': "Biện pháp canh tác: Tuyệt đối tránh trồng táo gần khu vực có cây bách xù (ký chủ trung gian của nấm rỉ sắt). Cắt bỏ các u bướu gỉ sắt trên cây bách xù vào mùa đông. Biện pháp hóa học: Sử dụng thuốc trừ nấm chứa hoạt chất Myclobutanil theo chỉ dẫn lâm sàng.",
    'Apple - healthy': "Chế độ chăm sóc: Cây táo đang sinh trưởng khỏe mạnh. Tiếp tục duy trì chế độ tưới nước nhỏ giọt, bón phân hữu cơ kết hợp NPK cân đối theo từng giai đoạn ra hoa, đậu quả và chủ động theo dõi sâu hại định kỳ.",
    'Blueberry - healthy': "Chế độ chăm sóc: Cây việt quất phát triển tốt. Cần duy trì nghiêm ngặt độ pH của đất ở mức axit (khoảng 4.5 - 5.5), bổ sung lớp phủ mùn hữu cơ (mulch) quanh gốc để giữ ẩm và cung cấp đủ ánh sáng mặt trời.",
    'Cherry (including sour) - Powdery mildew': "Biện pháp canh tác: Tăng cường cắt tỉa tạo tán để luồng không khí lưu thông tốt, tuyệt đối tránh tưới nước trực tiếp lên phiến lá vào buổi chiều tối. Biện pháp hóa học: Phun dầu Neem, dung dịch lưu huỳnh hoặc thuốc trừ nấm chuyên dụng khi phát hiện lớp phấn trắng đầu tiên.",
    'Cherry (including sour) - healthy': "Chế độ chăm sóc: Cây anh đào đạt trạng thái sinh lý tốt. Thực hiện cắt tỉa cành tăm, cành vượt hàng năm vào mùa ngủ đông để tạo tán, kết hợp bón phân gốc định kỳ giúp cây tích lũy dinh dưỡng cho vụ hoa sau.",
    'Corn (maize) - Common rust ': "Biện pháp canh tác: Thực hiện luân canh cây trồng khác họ, ưu tiên sử dụng các giống ngô lai có đặc tính kháng rỉ sắt. Biện pháp hóa học: Kịp thời phun thuốc trừ nấm chứa hoạt chất Propiconazole hoặc Azoxystrobin nếu mật độ vết bệnh lây lan nhanh trước giai đoạn phun râu.",
    'Corn (maize) - Northern Leaf Blight': "Biện pháp canh tác: Tiến hành cày lật đất sâu sau vụ thu hoạch để tiêu diệt tàn dư nấm. Trồng với mật độ thưa, bón phân cân đối, chú trọng tăng cường lượng phân Kali để tăng sức đề kháng. Biện pháp hóa học: Phun thuốc diệt nấm kịp thời khi thời tiết mát và ẩm kéo dài.",
    'Corn (maize) - healthy': "Chế độ chăm sóc: Khóm ngô sinh trưởng đồng đều và sạch bệnh. Cần đảm bảo cung cấp lượng nước dồi dào, đặc biệt là trong giai đoạn phân hóa mầm hoa (phun râu, trổ cờ) và chủ động bón thúc đúng chu kỳ.",
    'Grape - Black rot': "Biện pháp canh tác: Cắt tỉa triệt để các cành, lá và chùm quả bị nhiễm bệnh; thu dọn sạch sẽ xác bã thực vật dưới mương. Biện pháp hóa học: Thiết lập lịch phun thuốc trừ nấm định kỳ bảo vệ cây, đặc biệt là giai đoạn trước mùa mưa và khi nụ hoa bắt đầu nở.",
    'Grape - Esca (Black Measles)': "Biện pháp canh tác: Đây là bệnh hoại tử thân rễ phức tạp chưa có thuốc trị hóa học đặc hiệu. Cần cắt bỏ sâu phần gỗ bị hoại tử, bôi keo liền sẹo. Phải khử trùng dụng cụ cắt tỉa bằng cồn để tránh lây chéo, hạn chế tối đa việc gây vết thương cơ giới cho cây.",
    'Grape - Leaf blight (Isariopsis Leaf Spot)': "Biện pháp canh tác: Buộc dây, nâng giàn và cắt tỉa cành lá rậm rạp để cải thiện độ thông thoáng không khí, giảm độ ẩm đọng trên lá. Biện pháp hóa học: Can thiệp bằng các loại thuốc trừ nấm chứa gốc đồng hoặc Mancozeb ngay khi thời tiết chuyển sang mưa dầm.",
    'Grape - healthy': "Chế độ chăm sóc: Vườn nho duy trì trạng thái khỏe mạnh. Tiếp tục thực hiện các biện pháp quản lý thảm thực vật, làm sạch cỏ dại quanh gốc, bón phân vi lượng định kỳ và kiểm soát độ ẩm của đất để tránh thối rễ.",
    'Orange - Haunglongbing (Citrus greening)': "Biện pháp canh tác: Bệnh vàng lá gân xanh (HLB) không thể chữa khỏi và lây truyền qua rầy chổng cánh. Cần đào bỏ và tiêu hủy bằng cách đốt các cây đã nhiễm bệnh nặng. Biện pháp hóa học: Quản lý và tiêu diệt triệt để véc-tơ truyền bệnh (rầy chổng cánh) bằng thuốc trừ sâu đặc trị.",
    'Peach - Bacterial spot': "Biện pháp canh tác: Lựa chọn và chuyển đổi sang các giống đào có khả năng kháng khuẩn. Hạn chế tối đa hệ thống tưới phun mưa lên lá, đảm bảo lá cây luôn khô ráo nhanh nhất sau các đợt mưa. Biện pháp hóa học: Phun hợp chất kháng khuẩn gốc đồng hoặc oxytetracycline vào đầu mùa.",
    'Peach - healthy': "Chế độ chăm sóc: Cây đào không có dấu hiệu nhiễm bệnh. Cần chú ý bón lót phân hữu cơ hoai mục định kỳ, theo dõi độ thoát nước của đất và thực hiện các biện pháp rải bả phòng trừ ruồi vàng đục quả khi vào mùa sinh sản.",
    'Pepper, bell - Bacterial spot': "Biện pháp canh tác: Tuyệt đối không tưới nước rưới lên màng lá, nên dùng hệ thống tưới nhỏ giọt. Bắt buộc luân canh với các loại cây họ hòa thảo từ 2-3 năm, luôn sử dụng nguồn hạt giống đã được chứng nhận sạch bệnh.",
    'Pepper, bell - healthy': "Chế độ chăm sóc: Quần thể ớt chuông đang phát triển xanh tốt. Duy trì độ ẩm đất ổn định, tránh tình trạng úng ngập cục bộ và thường xuyên bổ sung canxi để phòng ngừa chứng thối đít quả (Blossom end rot).",
    'Potato - Early blight': "Biện pháp canh tác: Bổ sung lượng lớn phân hữu cơ và phân Kali để tăng sức đề kháng. Bắt buộc luân canh với các loài cây không thuộc họ Cà. Biện pháp hóa học: Chủ động phun thuốc trừ nấm (như Chlorothalonil) sớm ngay khi xuất hiện các đốm vòng nâu đầu tiên.",
    'Potato - Late blight': "Biện pháp canh tác: Bệnh úa muộn lây lan sinh lý cực kỳ nhanh, cần nhổ bỏ và tiêu hủy lập tức các cụm cây nhiễm bệnh. Giữ cho màng lá khô ráo. Biện pháp hóa học: Phun dập dịch khẩn cấp bằng các loại thuốc trừ nấm gốc đồng hoặc hoạt chất Mancozeb theo liều lượng khuyến cáo.",
    'Potato - healthy': "Chế độ chăm sóc: Cánh đồng khoai tây phát triển đạt chuẩn. Đảm bảo luống đất luôn tơi xốp, hệ thống rãnh thoát nước hoạt động tốt để củ phát triển tối đa, kết hợp vun xới kịp thời để củ không bị phơi nắng hóa xanh.",
    'Raspberry - healthy': "Chế độ chăm sóc: Cây phúc bồn tử sinh trưởng ổn định. Tiến hành cắt tỉa sát gốc các cành già yếu ngay sau vụ thu hoạch để dồn dinh dưỡng nuôi cành tơ, đồng thời cố định cành lên giàn để tránh gãy đổ.",
    'Soybean - healthy': "Chế độ chăm sóc: Nương đậu nành phát triển khỏe mạnh. Chủ động theo dõi sát sao mật độ các loại côn trùng chích hút và sâu ăn lá, đặc biệt chú ý bảo vệ cây vào giai đoạn hình thành hoa và tạo quả đậu.",
    'Squash - Powdery mildew': "Biện pháp canh tác: Điều chỉnh mật độ luống trồng thưa hơn để tăng cường ánh sáng mặt trời chiếu trực tiếp và độ thông gió. Biện pháp hóa học: Phun phòng ngừa bằng dung dịch Baking soda pha loãng, dầu Neem hoặc thuốc trừ nấm chứa lưu huỳnh khi phát hiện nấm phấn trắng.",
    'Strawberry - Leaf scorch': "Biện pháp canh tác: Thường xuyên dọn dẹp và cắt bỏ các lá già, lá có dấu hiệu cháy hoại tử. Tránh để lớp đất xung quanh gốc quá ẩm ướt trong thời gian dài, nên sử dụng màng phủ nông nghiệp hoặc rơm rạ khô để lót quả.",
    'Strawberry - healthy': "Chế độ chăm sóc: Luống dâu tây sinh trưởng tốt, không có dấu hiệu nấm mốc. Định kỳ thay mới lớp phủ đất (mulch) hàng năm, kiểm soát nghiêm ngặt lượng phân đạm bón vào và duy trì tưới nhỏ giọt quanh gốc.",
    'Tomato - Bacterial spot': "Biện pháp canh tác: Không thực hiện các thao tác cắt tỉa hay thu hoạch khi màng lá cà chua còn ướt sương hoặc sau mưa. Thực hiện luân canh cây trồng nghiêm ngặt, ngâm xử lý hạt giống bằng nước nóng (khoảng 50 độ C trong 25 phút) trước khi gieo.",
    'Tomato - Early blight': "Biện pháp canh tác: Chủ động cắt tỉa toàn bộ cành lá dưới thấp (gần mặt đất) để tránh các bào tử nấm từ đất bắn lên lá khi mưa. Biện pháp hóa học: Can thiệp bằng thuốc trừ nấm chứa hoạt chất Chlorothalonil hoặc gốc đồng ngay khi thấy đốm nâu đồng tâm.",
    'Tomato - Late blight': "Biện pháp canh tác: Bệnh mốc sương lây lan với tốc độ hủy diệt trong khí hậu mát và độ ẩm cao. Cần cách ly và tiêu hủy cây bệnh ngay lập tức. Biện pháp hóa học: Phun thuốc phòng ngừa gốc đồng hoặc Metalaxyl liên tục để bảo vệ phần cây chưa nhiễm bệnh.",
    'Tomato - Leaf Mold': "Biện pháp canh tác: Mở rộng cửa thông gió, lắp quạt đảo gió nếu trồng trong nhà kính để giảm triệt để độ ẩm đọng. Tuyệt đối hạn chế hệ thống tưới phun mưa rải trực tiếp lên lá, chỉ nên tưới thẩm thấu tại màng rễ.",
    'Tomato - Septoria leaf spot': "Biện pháp canh tác: Phát hiện và cắt tỉa ngay các lá có đốm nhỏ viền đen, dọn sạch tàn dư dưới luống, tránh tưới nước lên lá. Biện pháp hóa học: Sử dụng ngay các hợp chất diệt nấm gốc đồng hoặc thuốc trừ nấm phổ rộng khi bệnh mới xuất hiện ở lá chân.",
    'Tomato - Target Spot': "Biện pháp canh tác: Duy trì khoảng cách không gian đạt chuẩn giữa các khóm cây, cung cấp đủ ánh sáng. Thu gom và dọn dẹp sạch sẽ tàn dư thực vật, cỏ dại trong và sau mùa thu hoạch để cắt đứt chu kỳ sinh trưởng của nấm.",
    'Tomato - Tomato Yellow Leaf Curl Virus': "Biện pháp canh tác: Bệnh xoăn lá virus không có thuốc đặc trị. Phải tập trung kiểm soát chặt chẽ véc-tơ truyền bệnh là bọ phấn trắng bằng lưới che côn trùng hạt mịn, bẫy dính màu vàng và chọn mua các giống cà chua lai có đặc tính kháng virus.",
    'Tomato - Tomato mosaic virus': "Biện pháp canh tác: Virus khảm lây nhiễm chéo cơ học rất mạnh qua tay người làm vườn và công cụ cắt tỉa. Bắt buộc rửa tay bằng xà phòng, khử trùng công cụ thường xuyên. Tuyệt đối không hút thuốc lá gần khu vực trồng do virus có thể lây từ sợi thuốc lá.",
    'Tomato - healthy': "Chế độ chăm sóc: Vườn cà chua đang phát triển đạt năng suất. Tiếp tục duy trì chế độ dinh dưỡng ổn định với hàm lượng Kali và Canxi cao khi cây ra quả, tỉa chồi nách thường xuyên để tập trung dinh dưỡng nuôi thân chính."
}
# LOAD MODEL
@st.cache_resource
def load_models():
    return {
        "cnn": keras.models.load_model("custom_cnn_plant.keras", compile=False),
        "mobile": keras.models.load_model("mobilenet_v2_plant.keras", compile=False),
        "eff": keras.models.load_model("efficientnet_b0_plant.keras", compile=False)
    }

models = load_models()

# GIAO DIỆN
uploaded_file = st.file_uploader("Tải ảnh lá cây cần chẩn đoán...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Ảnh đã tải', use_container_width=True)
    
    if st.button('Dự đoán'):
        with st.spinner('Đang phân tích...'):
            # Tiền xử lý
            img_resized = img.resize((224, 224))
            img_array = keras.utils.img_to_array(img_resized) / 255.0
            img_tensor = np.expand_dims(img_array, axis=0)
            
            # Dự đoán
            p1 = models["cnn"].predict(img_tensor, verbose=0)
            p2 = models["mobile"].predict(img_tensor, verbose=0)
            p3 = models["eff"].predict(img_tensor, verbose=0)
            
            # Ensemble (1:5:2)
            ensemble_prob = (1.0 * p1 + 5.0 * p2 + 2.0 * p3) / 8.0
            
            idx = np.argmax(ensemble_prob)
            conf = np.max(ensemble_prob) * 100
            
            # Định nghĩa biến hiển thị dựa trên nhãn tìm được
            predicted_class_display = CLASS_NAMES[idx].replace('___', ' - ').replace('_', ' ')
            
            # Hiển thị kết quả
            st.success(f"### Kết quả: **{predicted_class_display}**")
            st.info(f"Độ tự tin: {conf:.2f}%")
            
            # Hiển thị lời khuyên
            remedy = REMEDIES.get(predicted_class_display, "Chưa có hướng dẫn cụ thể cho bệnh này. Vui lòng liên hệ chuyên gia.")
            st.warning(remedy)
