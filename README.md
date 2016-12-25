# Automated Rectification of Image

Implements the modified version of the following paper:  

[Chaudhury, Krishnendu, Stephen DiVerdi, and Sergey Ioffe. "Auto-rectification
of user photos." 2014 IEEE International Conference on Image Processing (ICIP).
 IEEE, 2014.](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42532.pdf)

Modifcation note: Instead of finding edge direction using structural tensor and its eigenvectors as in paper, I have used more reliable canny edge detection and probabalistic hough line transform.

##  Results

Input image:

![Input Image](/results/shelf.jpg)

After rectification:

![Rectified Image](/results/shelf_warped.png)

## How it works

First, compute list of 'edgelets'. An edgelet is a tuple of edge location, edge direction and edge strength. 

```python
edgelets1 = compute_edgelets(image)
vis_edgelets(image, edgelets1) # Visualize the edgelets
```

![Edgelets](/results/edgelets.png)

Next, find dominant vanishing point using ransac algorithm. In our case it turns out to be horizontal.

```python
vp1 = ransac_vanishing_point(edgelets1, num_ransac_iter=2000, 
                             threshold_inlier=5)
vp1 = reestimate_model(vp1, edgelets1, threshold_reestimate=5)
vis_model(image, vp1) # Visualize the vanishing point model
```

![Horizontal Vanishing Point](/results/horizontal_vp.png)

Remove the inliers for horizontal vanishing point. Vertical lines should now be dominant. Recompute the vanishing point using ransac should give us vertical vanishing point. 

```python
edgelets2 = remove_inliers(vp1, edgelets1, 10)
vp2 = ransac_vanishing_point(edgelets2, num_ransac_iter=2000,
                             threshold_inlier=5)
vp2 = reestimate_model(vp2, edgelets2, threshold_reestimate=5)
vis_model(image, vp2) # Visualize the vanishing point model
```

![Vertical Vanishing Point](/results/vertical_vp.png)

Finally, compute homography and warp the image so that we have a fronto parellel view with orthogonal axes: 

```
warped_img = compute_homography_and_warp(image, vp1, vp2,
                                         clip_factor=clip_factor)
```

![Rectified Image](/results/shelf_warped.png)