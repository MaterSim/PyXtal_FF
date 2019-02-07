def make_js(twojmax, diagonal):
    js = []
    for j1 in range(0, twojmax + 1):
        if diagonal == 2:
            js.append([j1, j1, j1])
        elif diagonal == 1:
            for j in range(0, min(twojmax, 2 * j1) + 1, 2):
                js.append([j1, j1, j])
        elif diagonal == 0:
            for j2 in range(0, j1 + 1):
                for j in range(j1 - j2, min(twojmax, j1 + j2) + 1, 2):
                    js.append([j1, j2, j])
        elif diagonal == 3:
            for j2 in range(0, j1 + 1):
                for j in range(j1 - j2, min(twojmax, j1 + j2) + 1, 2):
                    if j >= j1:
                        js.append([j1, j2, j])
    return js

print(len(make_js(6, 3)))
