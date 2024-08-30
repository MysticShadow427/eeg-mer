# meiju2025
<hr>
## To-do
- what I am thinking is to associate the vqvae codebook with the uique persons in the dataset but i will not use reconstruction loss I woud like to use only the classifcation loss and the other person specific losses,less what happens.
- we need to check the shape of the data and then accordingly make dataloader with reference of the eeg-conformer repo.
- check braindecode repo for another implementation which takes 3d input [bs,nc,nt].
- we can pass cwt features too the conformer (modify the dataloader accordingly and also augmentation for cwts), hence also add some freq based featurs too, we will experiment, but first experiment only with the og authors did in the eeg-conformer paper.
- maybe we decouple these tasks in task-1 and task-2.
    - SupCon + CodeBook for Task-1 (person identification) and SimClr + Multitask Learning for Task-2 (emotion classification).
