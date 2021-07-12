Use integrated Datasets
=======================

We built in a few datasets that can be used out-of-the-box, namely CLEVR, GQA, TextVQA and VQA2.
These ``PyTorch`` datasets can used to load any dataset that follows the same structure or format.
As an example, the OK-VQA dataset can be loaded using our ``VQADataset`` , and the GQA-OOD using our ``GQADataset`` .

.. code-block:: python

    from vqa_benchmarking_backend.datasets.GQADataset import GQADataset
    from vqa_benchmarking_backend.datasets.TextVQADataset import TextVQADataset
    from vqa_benchmarking_backend.datasets.VQADataset import VQADataset
    from vqa_benchmarking_backend.datasets.CLEVRDataset import CLEVRDataset

    # insert required paths

    # Vanilla GQA dataset
    gqa_dataset = GQADataset(question_file=qsts_path, img_dir= img_dir, img_feat_dir='', idx2ans=all_indices, name='GQA')

    # GQA-OOD splits
    gqa_dataset_odd_all = GQADataset(question_file=gqa_ood_testdev_all, img_dir= img_dir, img_feat_dir='', idx2ans=all_indices, name='GQA-OOD-ALL')
    gqa_dataset_odd_head = GQADataset(question_file=gqa_ood_testdev_head, img_dir= img_dir, img_feat_dir='', idx2ans=all_indices, name='GQA-OOD-HEAD')
    gqa_dataset_odd_tail = GQADataset(question_file=gqa_ood_testdev_tail, img_dir= img_dir, img_feat_dir='', idx2ans=all_indices, name='GQA-OOD-TAIL')

    # TextVQA dataset
    textvqa_dataset = TextVQADataset(question_file=text_vqa_qsts_path, img_dir=text_vqa_imgs_path, img_feat_dir='', idx2ans=all_indices)

    # CLEVR dataset
    clevr_dataset = CLEVRDataset(question_file=clevr_qsts_path, img_dir=clevr_img_dir, img_feat_dir='', idx2ans=all_indices)

    # Vanilla VQA2 dataset
    vqav2_dataset = VQADataset(
        val_question_file=vqav2_qsts_path, 
        val_annotation_file=vqav2_anno_path,
        answer_file=all_indices,
        img_dir=vqav2_img_dir,
        name='VQA2'
    )

    # OK-VQA using VQADataset
    okvqa_dataset = VQADataset(
	 	val_question_file=ok_vqa_qsts_path, 
        val_annotation_file=ok_vqa_anno_path,
	 	answer_file=all_indices,
	 	img_dir=ok_vqa_imgs_path,
        name='OK-VQA',
        dataset_fraction=1.0
    )
