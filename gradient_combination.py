output = []

def combination(n):
    if (n == 1):
        a = [True]
        b = [False]
        output.append(a)
        output.append(b)
        return output

    else:
        c = [True]
        d = [False]
        prev_binary_combination = combination(n - 1)
        binary_combination = []
        for i in prev_binary_combination:
            tmp = i + c
            binary_combination.append(tmp)
            tmpp = i + d
            binary_combination.append(tmpp)
        return binary_combination
