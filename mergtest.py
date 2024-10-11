import os
import cv2
import torch
import yt_dlp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from config import TEMPORAL
from model import get_model


class VideoFrameDataset(Dataset):
    def __init__(self, frame_folder, transform=None):
        self.frame_folder = frame_folder
        self.frames = sorted(os.listdir(frame_folder))
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_path = os.path.join(self.frame_folder, self.frames[idx])
        image = Image.open(frame_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


def download_video(video_url):
    # Tùy chọn tải video
    ydl_opts = {
        'format': 'best',
        'outtmpl': 'input_video.mp4'
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    print("Video đã tải xong.")


def extract_frames():
    os.makedirs('video_frames', exist_ok=True)
    video_path = 'input_video.mp4'
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f'video_frames/frame_{frame_count:04d}.jpg', frame)
        frame_count += 1

    cap.release()
    print("Đã tách video thành các khung hình.")


def create_gaussian_map(size=368):
    """Tạo Gaussian map cho center map"""
    x = torch.arange(0, size, 1, dtype=torch.float32)
    y = torch.arange(0, size, 1, dtype=torch.float32)

    mean = size // 2
    sigma = size // 4

    y = y.unsqueeze(1)
    gaussian = torch.exp(-((x - mean) ** 2 + (y - mean) ** 2) / (2 * sigma ** 2))
    gaussian = gaussian.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    return gaussian


def process_frames():
    # Định nghĩa transform với kích thước 368x368 (kích thước yêu cầu của mô hình)
    transform = transforms.Compose([
        transforms.Resize((368, 368)),
        transforms.ToTensor(),
    ])

    # Tạo DataLoader
    dataset = VideoFrameDataset('video_frames', transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Khởi tạo thiết bị và mô hình
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(TEMPORAL, device)

    # Load model weights
    checkpoint = torch.load('model_weightstop.pth',
                            map_location=device,
                            weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    # Tạo center map với kích thước phù hợp
    center_map = create_gaussian_map(368).to(device)

    with torch.no_grad():
        for i, frame in enumerate(dataloader):
            # Đảm bảo frame có kích thước đúng
            frame = frame.to(device)

            # Tạo temporal input bằng cách lặp lại frame
            temporal_frames = frame.repeat(1, TEMPORAL, 1, 1)

            # Dự đoán
            pred_heatmaps = model(temporal_frames, center_map)

            # Hiển thị hoặc xử lý kết quả
            print(f'Đã dự đoán khung hình {i}')


def main():
    video_url = 'https://www.youtube.com/watch?v=yvqYk1bZpr0'
    download_video(video_url)
    extract_frames()
    process_frames()


if __name__ == "__main__":
    main()