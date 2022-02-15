def bresenham(start, end):
    x1,y1 = start
    x2,y2 = end
    
    if abs(x2 - x1) < abs(y2 - y1): 
        # highline, one pixel on each y val because y is steeper than x
        if y2 > y1:
            return [(b,a) for (a,b) in lowline(y1,x1, y2,x2)]
        return [(b,a) for (a,b) in lowline(y2,x2, y1,x1)[::-1]]
    
    # lowline, one pixel on each x val because x is steeper than y
    if x2 > x1:
        return lowline(x1,y1, x2,y2)
    return lowline(x2,y2, x1,y1)[::-1]


def lowline(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1

    # increasing or decreasing y
    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy

    di = dy*2 - dx
    y = y1
    res = []
    for x in range(x1, x2+1):
        res.append((x,y))
        if di > 0:
            y = y + yi
            di = di - 2 * dx
        di = di + 2 * dy

    return res