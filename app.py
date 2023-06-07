import torch
import gradio as gr


def greet(name):
  return "Hello " + name + "!"



#read GPU info
ngpu=torch.cuda.device_count()
gpu_infos=[]
if(torch.cuda.is_available()==False or ngpu==0):if_gpu_ok=False
else:
    if_gpu_ok = False
    for i in range(ngpu):
        gpu_name=torch.cuda.get_device_name(i)
        if("MX"in gpu_name):continue
        if("10"in gpu_name or "16"in gpu_name or "20"in gpu_name or "30"in gpu_name or "40"in gpu_name or "A50"in gpu_name.upper() or "70"in gpu_name or "80"in gpu_name or "90"in gpu_name or "M4"in gpu_name or "T4"in gpu_name or "TITAN"in gpu_name.upper()):#A10#A100#V100#A40#P40#M40#K80
            if_gpu_ok=True#至少有一张能用的N卡
            gpu_infos.append("%s\t%s"%(i,gpu_name))
gpu_info="\n".join(gpu_infos)if if_gpu_ok==True and len(gpu_infos)>0 else "很遗憾您这没有能用的显卡来支持您训练"
gpus="-".join([i[0]for i in gpu_infos])



app = gr.Blocks()
with app:
  gr.Markdown(value = """# Medical Image Denoising-webui v.0.0.1\n
                          A user freindly webui for medical image denoising research\n
                          author: Guanghang Chen\n 
                          license: MIT""")
  with gr.Tabs():
    with gr.TabItem("Preprocess"):
      gr.Markdown(value = "## Preprocess")
      with gr.TabItem("upload slices"):
        preprocess_upload_slices = gr.Files(label="Upload slices", file_type="DCM", description="Upload slices in DCM format")
      precess = gr.Button("Preprocess")
    with gr.TabItem("Train"):
      gr.Markdown(value = "## Training")
    with gr.TabItem("Inference"):
      gr.Markdown(value = "## Inference")
      with gr.Row():
        choice_model = gr.Dropdown(label="Model", choices=["DnCNN", "UNet", "ResNet", "DnCNN-UNet", "DnCNN-ResNet", "UNet-ResNet", "DnCNN-UNet-ResNet"])
        choice_ckpt = gr.Dropdown(label="Checkpoint", choices=["ckpt1", "ckpt2", "ckpt3"])
      gr.Button("Inference")
    with gr.TabItem("Analysis"):
      gr.Markdown(value = "## Analysis")

app.launch()