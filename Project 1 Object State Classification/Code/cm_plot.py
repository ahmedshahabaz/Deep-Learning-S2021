    from plotcm import plot_confusion_matrix
    from sklearn.metrics import confusion_matrix

    # all_target = ground truth
    # all_pred = prediction of the model
    cm  = confusion_matrix(all_targets, all_preds)
    print(cm)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))

    #class names
    plot_confusion_matrix(cm, ["creamy_paste","diced","floured","grated","juiced","jullienne","mixed",
    "other","peeled","sliced","whole"])
    plt.show()
        # you can save the model here at specific epochs (ckpt) to load and evaluate the model on the val set

    print ()