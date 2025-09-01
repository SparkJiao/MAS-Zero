mat = [[1,0,1],[1,1,0],[1,1,0]]
m = len(mat)
n = len(mat[0])


height = [[0] * n for _ in range(m)]
for i in range(m):
    if i == 0:
        for j in range(n):
            height[i][j] = mat[i][j]
        continue
    for j in range(n):
        if mat[i][j] == 1:
            height[i][j] = height[i - 1][j] + 1

res = 0
for i in range(m):
    queue = [(height[i][0], 0)]
    row_res = 0
    for j in range(1, n):
        print(i, j)
        tmp = queue[-1]
        while tmp[0] >= height[i][j] and len(queue) > 1:
            queue = queue[:-1]
            tmp = queue[-1]

        if tmp[0] < height[i][j]:
            last_h, pos = tmp
            row_res += height[i][j]
            row_res += last_h * (j - pos)
        else:
            row_res += height[i][j] * 1
        queue.append((height[i][j], j))
        res += row_res

print(res)

