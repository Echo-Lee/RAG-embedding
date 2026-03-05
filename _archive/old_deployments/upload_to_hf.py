"""
上传 FAISS 索引到 HuggingFace Hub

使用方法：
1. 在 Colab 中运行此脚本
2. 输入你的 HF Token
3. 自动上传索引到 HF Hub
"""

# 安装依赖
print("📦 Installing dependencies...")
import subprocess
subprocess.run(["pip", "install", "-q", "huggingface_hub"], check=True)

from huggingface_hub import login, HfApi
import os

# ========== 配置 ==========
YOUR_HF_USERNAME = "YOUR_USERNAME"  # ⚠️ 修改为你的 HF 用户名
REPO_NAME = "rag-indexes"  # HF Hub 仓库名

# 索引路径（Drive）
INDEX_BASE_PATH = "/content/drive/MyDrive/Epiq Project/pipeline/faiss_index"

# ========== 执行上传 ==========

def main():
    print("="*50)
    print("🚀 HuggingFace Hub 索引上传工具")
    print("="*50)

    # 1. 登录
    print("\n📝 Step 1: Login to HuggingFace Hub")
    print("请访问: https://huggingface.co/settings/tokens")
    print("创建一个 Write token，然后粘贴到下面：")

    hf_token = input("\n🔑 Enter your HF Token: ").strip()

    if not hf_token:
        print("❌ Token 不能为空！")
        return

    try:
        login(token=hf_token)
        print("✅ 登录成功！")
    except Exception as e:
        print(f"❌ 登录失败: {e}")
        return

    # 2. 创建 Repository
    print(f"\n📦 Step 2: Creating repository '{YOUR_HF_USERNAME}/{REPO_NAME}'")

    api = HfApi()
    repo_id = f"{YOUR_HF_USERNAME}/{REPO_NAME}"

    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=False,
            exist_ok=True  # 如果已存在，不报错
        )
        print(f"✅ Repository 已创建/已存在")
        print(f"   URL: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"⚠️  创建失败（可能已存在）: {e}")

    # 3. 上传索引
    datasets = ['hospital', 'corruption']

    for dataset in datasets:
        dataset_path = os.path.join(INDEX_BASE_PATH, dataset)

        if not os.path.exists(dataset_path):
            print(f"\n⚠️  跳过 {dataset}: 路径不存在 ({dataset_path})")
            continue

        print(f"\n📤 Step 3.{datasets.index(dataset)+1}: Uploading {dataset} index")
        print(f"   Source: {dataset_path}")
        print(f"   Target: {repo_id}/{dataset}")

        try:
            api.upload_folder(
                folder_path=dataset_path,
                repo_id=repo_id,
                repo_type="dataset",
                path_in_repo=dataset
            )
            print(f"✅ {dataset} 上传成功！")
        except Exception as e:
            print(f"❌ {dataset} 上传失败: {e}")

    # 4. 完成
    print("\n" + "="*50)
    print("🎉 上传完成！")
    print("="*50)
    print(f"\n查看你的数据: https://huggingface.co/datasets/{repo_id}")
    print("\n下一步：")
    print("1. 访问 https://huggingface.co/spaces")
    print("2. 创建新 Space，选择 Gradio SDK")
    print("3. 复制 gradio_app.py 的内容到 Space 的 app.py")
    print("4. 修改 YOUR_HF_USERNAME 为你的用户名")
    print("5. 等待构建完成，访问你的 Space URL！")

if __name__ == "__main__":
    # 检查配置
    if YOUR_HF_USERNAME == "YOUR_USERNAME":
        print("❌ 错误: 请先修改 YOUR_HF_USERNAME 为你的 HF 用户名！")
        print("   在脚本第 13 行修改")
    else:
        main()
