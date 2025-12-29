import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CelebADataset(Dataset):
    def __init__(self, folder_path, image_size=256):
        self.folder_path = folder_path
        # 1. 获取文件夹内所有 jpg 图片的文件名列表
        self.image_paths = [
            os.path.join(folder_path, f) 
            for f in os.listdir(folder_path) 
            if f.endswith('.jpg') or f.endswith('.png')
        ]
        
        # 2. 定义预处理 (Transforms)
        self.transform = transforms.Compose([
            transforms.Resize(image_size),           # 确保大小正确
            transforms.CenterCrop(image_size),       # 防止图片不是正方形
            transforms.ToTensor(),                   # [0, 255] int -> [0, 1] float
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # A. 读取路径
        img_path = self.image_paths[idx]
        
        # B. 加载图片 (PIL 库自动处理 JPG 解压膨胀)
        image = Image.open(img_path).convert("RGB")
        
        # C. 应用变换
        image = self.transform(image)
        
        return image

DATASET_FOLDER = "/home/zehua/.cache/kagglehub/datasets/badasstechie/celebahq-resized-256x256/versions/1/celeba_hq_256/"

dataset = CelebADataset(DATASET_FOLDER)


import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()

    def forward(self, x, t):
        # 1. 第一次卷积
        h = self.bnorm1(self.relu(self.conv1(x)))
        # 2. 注入时间信息 (关键点！)
        time_emb = self.relu(self.time_mlp(t))
        # 将时间向量扩展到 (Batch, Channel, 1, 1) 形状并加到特征图上
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        # 3. 第二次卷积 & 下采样/上采样
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)

class SimpleUNet(nn.Module):
    """
    为了演示清晰，这是一个简化版的 UNet。
    如果要跑 256x256，通常还需要加入 Self-Attention 层。
    """
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3
        time_emb_dim = 32

        # 时间编码层
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.init_conv = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # 下采样路径
        self.downs = nn.ModuleList([
            Block(down_channels[i], down_channels[i+1], time_emb_dim)
            for i in range(len(down_channels)-1)
        ])

        # 上采样路径
        self.ups = nn.ModuleList([
            Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True)
            for i in range(len(up_channels)-1)
        ])

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # 1. 计算时间嵌入
        t = self.time_mlp(timestep)
        x = self.init_conv(x)

        # 2. Encoder (Down)
        residuals = []
        for down in self.downs:
            x = down(x, t)
            residuals.append(x)

        # 3. Decoder (Up)
        for up in self.ups:
            residual = residuals.pop()
            # 拼接 Skip Connection
            x = torch.cat((x, residual), dim=1)
            x = up(x, t)

        return self.output(x)


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        # 1. 预计算调度表 (Schedule)
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0) # 累乘 alpha

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        """
        前向过程：直接算出 t 时刻的加噪图
        x_t = sqrt(alpha_hat) * x_0 + sqrt(1 - alpha_hat) * epsilon
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        """为 Batch 中的每张图随机选一个时间步 t"""
        return torch.randint(low=1, high=self.noise_steps, size=(n,), device=self.device)

    def sample(self, model, n):
        """
        核心逆向采样过程
        n: 想生成的图片数量
        """
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            # 1. 从纯高斯噪声开始 (x_T)
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)

            # 2. 倒序循环：从 T=1000 到 t=1
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device) # 创建时间步 tensor

                # 3. 预测噪声
                predicted_noise = model(x, t)

                # 4. 获取当前时刻的 alpha 值
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                # 5. 核心公式：计算 x_{t-1}
                # 只有 t > 1 时才加随机噪声，最后一步 t=1 不加
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                # 公式推导结果 (Standard DDPM implementation)
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        # 6. 后处理：把 [-1, 1] 还原回 [0, 255]
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2 # 变回 [0, 1]
        x = (x * 255).type(torch.uint8)
        return x


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # 用于显示进度条
import logging

# 假设之前的类代码保存在这些文件中
# 如果你把所有代码都在一个文件里，直接用即可
# from modules import SimpleUNet, Diffusion, CelebADataset

# ==========================================
# 1. 配置与准备
# ==========================================
def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def train():
    # 核心超参数
    run_name = "DDPM_Uncond_CelebA"
    epochs = 300            # 在 A100 上，256x256 可能需要多跑一会
    batch_size = 128        # 4张 A100 显存很大，可以设大一点，比如 128 或 256
    image_size = 256
    # dataset_path = "./data/CelebAMask-HQ/CelebA-HQ-img" # 替换为你的实际路径
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 3e-4

    setup_logging(run_name)
    
    # ----------------------
    # 数据与模型加载
    # ----------------------
    dataset = CelebADataset(DATASET_FOLDER, image_size=image_size)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=32) 
    # num_workers 取决于 CPU 核心数，A100 服务器通常 CPU 很强

    print("正在初始化模型...")
    model = SimpleUNet().to(device)
    
    # === 关键：利用多卡并行 ===
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 张 GPU 进行训练!")
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=image_size, device=device)
    
    # ==========================================
    # 2. 训练循环
    # ==========================================
    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch}:")
        # pbar = tqdm(dataloader) # 进度条包装器
        
        # for i, (images, _) in enumerate(pbar): 
        for i, images in enumerate(dataloader): 
            images = images.to(device)
            
            # A. 采样时间步 t
            # 为 Batch 里的每张图随机选一个 t (1 到 1000)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            
            # B. 前向加噪 (Forward Process)
            # x_t: 加噪后的图, noise: 我们生成的真值噪声 (Ground Truth)
            x_t, noise = diffusion.noise_images(images, t)
            
            # C. 模型预测 (Predict)
            # 让模型看 x_t 和 t，猜 noise 长什么样
            predicted_noise = model(x_t, t)
            
            # D. 计算 Loss
            loss = mse(noise, predicted_noise)
            print(f'>> Loss in epoch: {epoch}, step: {i}: {loss}')
            
            # E. 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # pbar.set_postfix(MSE=loss.item())
        
        # ==========================================
        # 3. 保存与监控
        # ==========================================
        # 每 10 个 Epoch 保存一次模型权重
        if epoch % 10 == 0:
            # 注意：如果是 DataParallel，保存时要存 model.module.state_dict()
            save_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(save_model.state_dict(), os.path.join("models", run_name, f"ckpt_{epoch}.pt"))
            print(f"模型已保存至 epoch {epoch}")

            # (可选) 这里其实应该写一个采样函数来看看生成效果，暂时省略
            
import torch
from torchvision.utils import save_image
# 假设你的类定义在 modules.py，如果不是，请把 SimpleUNet 和 Diffusion 类粘贴过来
# from modules import SimpleUNet, Diffusion 

def inference():
    # 1. 设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_id = 0
    checkpoint_path = f"./models/DDPM_Uncond_CelebA/ckpt_{ckpt_id}.pt" # 换成你最新的权重文件路径
    
    # 2. 初始化模型
    model = SimpleUNet().to(device)
    
    # 3. 加载权重 (处理 DDP 的 module. 前缀)
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # 这里的逻辑是：如果权重字典里的 key 有 'module.'，我们把它去掉
    state_dict = ckpt
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") if "module." in k else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    print("模型加载成功！")

    # 4. 初始化采样器
    diffusion = Diffusion(img_size=256, device=device)
    
    # 5. 生成图片
    # 一次生成 8 张看看效果
    generated_images = diffusion.sample(model, n=8)
    
    # 6. 保存结果
    # save_image 会自动把 tensor 变成网格图
    # generated_images 是 [N, 3, H, W] 且是 0-255 的 uint8，需要转回 float 0-1 才能给 save_image 用
    save_image(generated_images.float() / 255.0, f"result_sample_{ckpt_id}.png")
    print("生成完成！请查看 result_sample.png")

if __name__ == "__main__":
    # train()
    inference()
