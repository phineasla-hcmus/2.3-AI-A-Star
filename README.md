# 2.3-AI-A-Star
A* and Alt Pathfinding Algorithm in Python  

## Challenge
Given a grayscale height map (`img/map.bmp`), starting point, goal point and `m` constant (`input.txt`), find the shortest path using A*.

### Notes and constraints
- The pixel value is the height of that pixel. 
- You can only move from a point to its adjacent neighbor if difference in height are not larger than `m`: ![constraint](https://latex.codecogs.com/svg.latex?\large&space;|x_1-x_2|\leq{1},|y_1-y_2|\leq{1},|\Delta{a}|\leq{m})  
- The cost from one point to another is as follow: 
![real_cost_equation](https://latex.codecogs.com/svg.latex?\large&space;\sqrt{(x_2-x_1)^2&plus;(y_2-y_1)^2}&plus;(1/2*sgn(\Delta{a})&plus;1)*|\Delta{a}|)
- File `input.txt` structure:
```
(x1;y1)
(x2;y2)
m constant
```
Example: 
```
(74;213)
(96;311)
10
```
