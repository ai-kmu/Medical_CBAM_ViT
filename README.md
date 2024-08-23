### Medical_CBAM_ViT
<h1>Leveraging Inductive Bias in ViT for Medical Image Diagnosis</a></h1>

PyTorch official implementation of (Jungmin Ha, Euihyun Yoon, Sungsik Kim, Jinkyu Kim, and Jaekoo Lee. "Leveraging Inductive Bias in ViT for Medical Image Diagnosis" BMVC, 2024).


## Description
![스크린샷 2024-08-23 오후 2 07 05](https://github.com/user-attachments/assets/ac6bcc6d-5b6a-4fea-abdc-0fa5da7588c6)

An overview of our proposed model. Built upon Vision Transformer, we use the following three building blocks: (1) Stem Block, (2) SWA Block for 1st and 2nd stages, and (3) DA Block for 3rd and 4th stages. In image classification, the output feature map undergoes Global Average Pooling(GAP) and MLP processing. For segmentation, fused feature maps with Fused Feature Pyramid Network(FPN) from Stages are utilized. (b, c, d) Detailed Explanation of Local Attenton, Shifted-Window Attention and Deformable Attention


![스크린샷 2024-08-23 오후 2 13 05](https://github.com/user-attachments/assets/a6a36abd-a0ad-487a-b4fd-373a5982ded4)
Comparison of classification and segmentation performance on various datasets. Note that scores in parenthesis represent results with the black-hat transform as preprocess- ing. Bold text indicates the best performance, while underlined text indicates the second-best performance among all models.

## Requirements
- PyTorch (> 1.2.0)
- torchvision
- numpy


