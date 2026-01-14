# download_model.py（仅用于下载，下载完成后可删除）
import os
import shutil
import time

# 1) 固定一个“稳定的缓存目录”（放在当前工作目录下，评测机一般允许写）
#    注意：MODELSCOPE_CACHE 最好在 import modelscope 之前设置（很多教程强调这一点）
cache_root = os.path.abspath("./.modelscope_cache")
os.makedirs(cache_root, exist_ok=True)
os.environ["MODELSCOPE_CACHE"] = cache_root

from modelscope import HubApi
from modelscope import snapshot_download

# ====== 按评测要求：token 写在代码里 ======
api = HubApi()

model_id = "Qwen/Qwen3-0.6B"
local_dir = "./local_model"

def rm_temp_dirs(root: str):
    for dirpath, dirnames, _ in os.walk(root):
        for d in list(dirnames):
            if d == "._____temp":
                shutil.rmtree(os.path.join(dirpath, d), ignore_errors=True)

def ensure_clean_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)

print(f"开始从 ModelScope 下载模型 {model_id} ...")
max_retry = 5

for attempt in range(1, max_retry + 1):
    try:
        # 每次重试都把输出目录清干净，避免半成品
        ensure_clean_dir(local_dir)
        rm_temp_dirs(local_dir)
        rm_temp_dirs(cache_root)

        downloaded_dir = snapshot_download(model_id=model_id, cache_dir=cache_root)

        shutil.copytree(downloaded_dir, local_dir, dirs_exist_ok=True)

        # 简单校验：至少要有权重文件/配置文件之一
        ok = False
        for root, _, files in os.walk(local_dir):
            for fn in files:
                if fn.endswith((".safetensors", ".bin", ".pt")) or fn in ("config.json", "configuration.json"):
                    ok = True
                    break
            if ok:
                break
        if not ok:
            raise RuntimeError("下载完成但未发现权重/配置文件，可能下载不完整。")

        print(f"下载完成！模型已保存到: {local_dir}")
        break

    except Exception as e:
        print(f"[第 {attempt}/{max_retry} 次] 下载失败：{type(e).__name__}: {e}")
        # 清理掉可能的半成品，再重试
        rm_temp_dirs(local_dir)
        rm_temp_dirs(cache_root)
        if attempt == max_retry:
            raise
        time.sleep(3 * attempt)

