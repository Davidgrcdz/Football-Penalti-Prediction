import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

def get_model_predictions(model, dataloader, device, model_type: str):
    """
    Función auxiliar universal para obtener las predicciones y etiquetas de cualquier modelo.
    Usa un parámetro explícito 'model_type' para decidir cómo llamar al modelo.
    """
    model.eval()
    all_labels, all_scores = [], []
    model_type = model_type.lower()

    with torch.no_grad():
        for batch in dataloader:
            # Modelos de secuencia (LSTM, TCN, etc.)
            if model_type in ['lstm', 'tcn']:
                inputs, lengths, labels = batch
                inputs, lengths = inputs.to(device), lengths.to(device)
                outputs = model(inputs, lengths)
            # Modelos de vector único (MLP)
            elif model_type == 'mlp':
                inputs, labels = batch
                inputs = inputs.to(device)
                outputs = model(inputs)
            else:
                raise ValueError(f"Tipo de modelo no reconocido en get_model_predictions: '{model_type}'")
            
            scores = torch.softmax(outputs, dim=1)
            all_labels.append(labels.cpu().numpy())
            all_scores.append(scores.cpu().numpy())

    if not all_labels:
        return np.array([]), np.array([]), np.array([])

    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)
    all_preds = np.argmax(all_scores, axis=1)
    
    return all_labels, all_preds, all_scores

def evaluate_model(model, dataloader, device, class_names, model_type: str):
    """
    Función universal para generar y mostrar la matriz de confusión,
    la curva ROC y la curva Precision-Recall para cualquier modelo.
    Además, imprime resúmenes numéricos de las métricas.
    """
    print(f"\n--- Iniciando Evaluación para modelo tipo: {model_type.upper()} ---")
    
    # Pasa el model_type a la función auxiliar
    y_true, y_pred, y_score = get_model_predictions(model, dataloader, device, model_type)
    
    if y_true.size == 0:
        print("⚠️ No hay datos en el dataloader para evaluar.")
        return

    num_classes = len(class_names)

    # --- 1. Matriz de Confusión ---
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    # --- Imprimir resumen de la Matriz de Confusión ---
    total_samples = cm.sum()
    correct_predictions = np.diag(cm).sum()
    incorrect_predictions = total_samples - correct_predictions
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    
    print("\n=== Resultados de la Matriz de Confusión ===")
    print(f"Aciertos (diagonal): {correct_predictions}")
    print(f"Errores (fuera de la diagonal): {incorrect_predictions}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # --- Dibujar la Matriz de Confusión ---
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Matriz de Confusión ({model_type.upper()})")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel("Etiqueta Real")
    plt.xlabel("Etiqueta Predicha")
    plt.tight_layout()
    plt.show()

    # --- 2. Curva ROC Multiclase ---
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    plt.figure(figsize=(6, 5))
    for i in range(num_classes):
        if i < y_true_bin.shape[1]:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label="Azar (AUC = 0.50)")
    plt.xlabel("Tasa de Falsos Positivos (FPR)")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
    plt.title(f"Curva ROC Multi-clase ({model_type.upper()})")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()

    # --- 3. Curva Precision-Recall Multiclase ---
    precision, recall, average_precision = dict(), dict(), dict()
    
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        if i < y_true_bin.shape[1]:
            precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
            average_precision[i] = average_precision_score(y_true_bin[:, i], y_score[:, i])
            plt.plot(recall[i], precision[i], lw=2, label=f'Clase \'{class_names[i]}\' (AP = {average_precision[i]:.3f})')
    
    # Micro-average
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_bin.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(y_true_bin, y_score, average="micro")

    plt.plot(recall["micro"], precision["micro"],
             label=f'Micro-promedio (AP = {average_precision["micro"]:.3f})',
             color='deeppink', linestyle=':', linewidth=4)
    
    # --- Imprimir resumen de la Curva PR ---
    macro_ap = np.mean([average_precision[i] for i in range(num_classes) if i in average_precision])
    print(f"\n=== Resultados de la Curva PR ===")
    print(f"Macro-promedio AP (PR-AUC): {macro_ap:.4f}")
    print(f"Micro-promedio AP (PR-AUC): {average_precision.get('micro', 0.0):.4f}")
    
    plt.xlabel("Recall (Sensibilidad)")
    plt.ylabel("Precisión")
    plt.title(f"Curva Precision-Recall Multiclase ({model_type.upper()})")
    plt.legend(loc="best")
    plt.grid(alpha=0.4)
    plt.show()