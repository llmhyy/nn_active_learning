def combination(n):
    output = []
    if n == 1:
        a = [True]
        b = [False]
        output.append(a)
        output.append(b)
        return output
    else:
        prev_binary_combination = combination(n - 1)
        binary_combination = []
        for i in prev_binary_combination:
            a = i + [True]
            binary_combination.append(a)
            b = i + [False]
            binary_combination.append(b)
        return binary_combination
