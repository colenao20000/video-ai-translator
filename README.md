# vTranslate Studio

Ứng dụng desktop hỗ trợ bản địa hóa video: trích xuất lời thoại và phụ đề, dịch, gán người nói, tạo lồng tiếng và xuất video đã biên tập.

## Chức năng

- Trích xuất lời thoại bằng ASR, quét phụ đề trên hình bằng OCR, hoặc đối chiếu hai nguồn.
- Chỉnh sửa văn bản, thời gian, nhân vật, giọng và sắc thái từng câu trước khi xuất.
- Ước lượng người nói bằng phân tích âm học cục bộ; các gợi ý cần được rà soát trước khi gán giọng.
- Tạo lồng tiếng tại máy bằng VieNeu-TTS v3 Turbo ONNX; clone giọng từ các mẫu WAV đi kèm và không cần API key.
- Căn chỉnh bản thu trên timeline, làm mờ hoặc che phụ đề gốc, chỉnh khung hình và logo, rồi xuất MP4 bằng FFmpeg.
- Lưu dự án để tiếp tục biên tập sau.

## Cài đặt

Cần Python 3.11 trở lên cùng FFmpeg và FFprobe có thể chạy từ terminal. Cài ứng dụng và các thư viện Python:

```bash
python -m pip install -e .
```

Để tách giọng và nhạc nền bằng Demucs, cài thêm gói tùy chọn nặng:

```bash
python -m pip install -e '.[separation]'
```

Demucs có thể tải trọng số mô hình trong lần chạy đầu. Nếu chưa cài được Demucs, ứng dụng dùng bộ lọc DSP xấp xỉ và sẽ đánh dấu kết quả đó; các rãnh dự phòng có thể còn lẫn giọng hoặc nhạc.

Chạy bản đã cài:

```bash
vtranslator
```

Hoặc chạy trực tiếp trong thư mục dự án:

```bash
python app.py
```

## Quy trình đề xuất

1. Mở video và chọn vùng OCR bao phủ phụ đề gốc.
2. Chạy nhận diện thoại hoặc OCR; kiểm tra câu chữ, thời gian và các dòng có độ tin cậy thấp.
3. Phân vai, nghe thử và xác nhận tên/giọng. Nếu không chắc người nói, giữ ở trạng thái chưa xác định cho đến khi chỉnh thủ công.
4. Dịch và hiệu đính bản dịch. Chỉnh sắc thái diễn xuất theo từng câu nếu cần.
5. Làm mờ hoặc che vùng phụ đề gốc, xem lại timeline và trạng thái các câu thoại.
6. Xuất video và mở lại dự án đã lưu khi cần.

## Dịch vụ và dữ liệu cần cấu hình

- FFmpeg và FFprobe xử lý media và kiểm tra video đầu ra; có thể đặt đường dẫn trong Cài đặt.
- Dịch thuật dùng dịch vụ tương thích OpenAI API, thường là 9router cục bộ; URL và API key cấu hình trong Cài đặt.
- Phân tích người nói ngoại tuyến chỉ là gợi ý từ đặc trưng âm học; cần xác nhận thủ công trước khi xuất nếu kết quả chưa chắc chắn.
- Mở **Cài đặt → Âm thanh & TTS** rồi bấm **Tải VieNeu-TTS v3 Turbo (ONNX)** để tải mô hình khoảng 640 MB một lần. Trọng số chính nằm trong `models/vieneu_models`, codec được lưu trong `models/huggingface_cache`. TTS chạy ngoại tuyến sau khi tải xong và không dùng khóa Google/Gemini.
- OCR/ASR có thể sai do chất lượng hình, tiếng ồn, giọng chồng lấn hoặc ngôn ngữ nguồn. Hãy rà lại nội dung trước khi xuất bản.

Khóa dịch vụ được lưu qua kho khóa hệ điều hành. Video nguồn, dự án và tệp trung gian được xử lý trên máy theo cấu hình ứng dụng.
