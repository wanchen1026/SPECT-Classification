# SPECT-Classification
code for *Deep Convolutional Neural Networks for Multi-Class Classification of Three-Dimensional Brain Images*
## train separately
apply different models by replacing *model* with the models in **model.py**
- train: specify the hospital name (CG/IS)
``python train.py --dir_name ./logs/_______ --hospital __``
- test:
``python train.py --dir_name ./logs/_______ --test``
## co-train
apply different models by replacing *model* with the models in **co_model.py**
- train:
``python co_train.py --dir_name ./logs/_______``
- test:
``python co_train.py --dir_name ./logs/_______ --test``
