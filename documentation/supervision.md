
Fine Tune Transformer



Next step 
Monte carlo MCMC Drop out approximation Gal. MC UQ

Guassian processes for uncertainty 

Dataset with Explanations


Presentation: Image 6 

Mean Confidence (0.166) = np.mean(pred_np) - This is the average prediction probability across all pixels

Overall Confidence (0.833) = 1 - overall_uncertainty - This is calculated from uncertainty methods






Metric	How Calculated	What It Means	Example
Max Confidence	np.max(probabilities)	Highest probability pixel	0.89 = 89% confident about best pixel
Mean Prediction	np.mean(probabilities)	Average prediction	0.16 = 16% of image predicted as polyp
Overall Confidence	1 - normalized_entropy	Model certainty about prediction	0.37 = 37% confident overall