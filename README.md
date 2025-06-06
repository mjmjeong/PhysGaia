# 🌱 PhysGaia: A Physics-Aware Dataset of Multi-Body Interactions for Dynamic Novel View Synthesis

## [Dataset](https://huggingface.co/datasets/mijeongkim/PhysGaia/tree/main) | [Project Page](https://cv.snu.ac.kr/research/PhysGaia/index.html) | arXiv Paper

 

## **⭐️ Key Highlights**

- **💥 Multi-body interaction**
- **💎 Various materials** across all modalities
    - Liquid, Gas, Viscoelastic substance, and Textile
- **✏️ Physical evaluation**
    - physics parameters
    - ground-truth 3D trajectories
- **😀 Research friendly!!**
    - Providing codes for recent Dynamic Novel View Synthesis (DyNVS) models
        - [Shape-of-motion](https://github.com/mjmjeong/PhysGaia/tree/SOM), [4DGS](https://github.com/mjmjeong/PhysGaia/tree/4DGS_hex), [STG](https://github.com/mjmjeong/PhysGaia/tree/spacetime), [D-3DGS](https://github.com/mjmjeong/PhysGaia/tree/deformable)
    - Supporting **diverse training setting**: both monocular & multiview reconstruction

## **🔥 Ideal for “Next” Research**

- 🧠 **Physical reasoning in dynamic scenes**
    - Offering ground-truth physics parameters for precise evaluation of inverse physics estimation
    - Offering ground-truth 3D trajectories for assessing actual motion beyond photorealism.
- 🤝 **Multi-body physical interaction modeling**
- 🧪 **Material-specific physics solver integration**
- 🧬 **Compatibility with existing DyNVS models**

## **📂 Dataset Structure**

Each folder is corresponding to each scene, containing the following files: 

```bash
{material_type}_{scene_name}.zip
│
├── render/                              # Rendered images
│   ├── train/                           # Images for training
│   └── test/                            # Images for evaluation
│
├── point_cloud.ply                      # COLMAP initialization (PatchMatch & downsampling)
├── camera_info_test.json                # Monocular camera info for test
├── camera_info_train_mono.json          # Monocular camera info for training
├── camera_info_train_multi.json         # Multi-view camera info for training
│
├── {scene_name}.hipnc                   # Houdini source file (simulation or scene setup)
├── particles/                           # Ground-truth trajectories
```

## **👩🏻‍💻 Code Implementation**

Please check each branch for integrated code for recent DyNVS methods. 

- **Shape-of-motion (arXiv 24.07):** https://github.com/mjmjeong/PhysGaia/tree/SOM
- **4DGS (CVPR 24):** https://github.com/mjmjeong/PhysGaia/tree/4DGS_hex
- **STG (CVPR 24):** https://github.com/mjmjeong/PhysGaia/tree/spacetime
- **D-3DGS (CVPR 24):** https://github.com/mjmjeong/PhysGaia/tree/deformable

## **💳 Citation**

```bash
TBD
```

## 🤝 Contributing

We welcome contributions to expand the dataset (additional modality for new downstream tasks, , implementation for other models, etc.)

Reach out via opening an issue/discussion in the repo.


## 🛠️ Future Plans

- Update fidelity of the generated scenes
- Add more easier scenes: providing more accessible starting points
- Add guidelines using Houdini source files: ex) How to obtain a flow field?

## **💳 License**

This project is released under the Creative Commons Attribution-NonCommercial 4.0 license.

*✅ Free to use, share, and adapt for non-commercial research*
