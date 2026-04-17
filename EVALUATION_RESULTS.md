# Model Evaluation Summary

## Overall Performance Metrics

After fixing label mapping issue (ground truth used label 7 for agricultural, model uses label 6):

- **Mean Pixel Accuracy**: 15.34%
- **Mean IoU (mIoU)**: 4.01%
- **Mean Dice Score**: 4.97%

## Per-Class Performance

| Class | IoU (%) | Dice (%) |
|-------|---------|----------|
| Background | 0.96 | 1.64 |
| Building | 2.61 | 4.73 |
| Road | 0.00 | 0.00 |
| Water | 0.00 | 0.00 |
| Barren | 0.89 | 1.59 |
| Forest | 0.07 | 0.13 |
| **Agricultural** | **23.58** | **26.72** |

## Key Findings

### Label Mismatch Issue
- The ground truth masks use label **7** for agricultural class
- The model was trained to predict label **6** for agricultural class  
- This caused a significant mismatch in evaluation

### Performance Analysis
1. **Agricultural class performs best** (23.58% IoU) after fixing the label mapping
2. **Road and Water classes have 0% IoU** - the model is not detecting these classes at all
3. **Overall low performance** suggests:
   - Model may not be fully trained (checkpoint is from epoch 29)
   - Possible data distribution issues
   - May need more training epochs or different hyperparameters

### Error Rates
Based on 50 validation samples:
- Mean pixel error rate: ~84.66% (100% - 15.34% accuracy)
- The model struggles with most classes except agricultural

## Visualization Files Generated
- ✓ per_class_metrics.png - Bar charts showing IoU and Dice scores per class

## Recommendations
1. Check if model was fully trained (epoch 29/30)
2. Investigate why road and water classes have 0% detection
3. Consider retraining with:
   - More epochs
   - Better data augmentation
   - Class balancing
4. Fix the label mapping in the dataset preparation phase
