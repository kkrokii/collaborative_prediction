For experiments on synthetic dataset, run:
    python simulation.py --output_file=output.pd --n1=50 --n2=50 --p=10 --beta_perturbation=1

For experiments on a real-world dataset, 
1. Save datasets in datasets_dir
2. Check column name of the reponse, response_column for instance.
3. Run:
    python clustering.py --output_file=output.pd --y_column=response_column--input_dir=datasets_dir

For experiments using neural networks,
1. Save datasets in datasets_dir
2. Check column name of the reponse, response_column for instance.
3. Run:
    python mlp.py --output_file=output.pd --y_column=response_column --input_dir=datasets_dir --output_model=model.pth 