#POSE COMPARISON AND POSE SCORE CALCULATION



#How to Use

-to get the results for squatting video keep the command same as below

``` 
python start_here2.py --activity "squat - front" --video1 "squat.mp4" --video2 "squat3.mp4" --lookup "squat_lookup.pickle" 
```



- to get the results for stretching video make the following changes
```
change the --activity to "stretch - side" for the stretching video
change the --video1 to "stretch.mp4"
change the --video2 to "stretch4.mp4" 
change the --lookup to "stretch_lookup.pickle" #or[YOUR_NAME].pickle
```


- to get the keypoints of the videos for comparison, run the following command

```
python keypoints_from_video.py --activity "stretch - side" --video "stretch4.mp4"  --lookup "lookup_stretch.pickle"
```


- to get keypoints for the squat video make the following changes
```
change the --activity to "squat - front"
change the --video to "squat.mp4"
change the --lookup to "squat_lookup.pickle" #or[YOUR_NAME].pickle
```

