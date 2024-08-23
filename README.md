### Medical_CBAM_ViT
<h1>Leveraging Inductive Bias in ViT for Medical Image Diagnosis</a></h1>

PyTorch official implementation of (Myunghak Lee, Wooseong Cho, Sungsik Kim, Jinkyu Kim, and Jaekoo Lee. "Distillation for High-Quality Knowledge
Extraction via Explainable Oracle Approach" BMVC, 2023).


## Description
![스크린샷 2024-08-23 오후 2 07 05](https://github.com/user-attachments/assets/ac6bcc6d-5b6a-4fea-abdc-0fa5da7588c6)

An overview of our proposed model. Built upon Vision Transformer, we use the following three building blocks: (1) Stem Block, (2) SWA Block for 1st and 2nd stages, and (3) DA Block for 3rd and 4th stages. In image classification, the output feature map undergoes Global Average Pooling(GAP) and MLP processing. For segmentation, fused feature maps with Fused Feature Pyramid Network(FPN) from Stages are utilized. (b, c, d) Detailed Explanation of Local Attenton, Shifted-Window Attention and Deformable Attention
