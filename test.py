result = []
for a in range(0, 76):
    for h in range(0, 76):
        for f in range(0, 3):
            result.append((a, h, f))

a = 1
h = 0
f = 0
index = a * 3 * 76 + h * 3 + f
print(index)
print(a, h, f)
print(result[index])
