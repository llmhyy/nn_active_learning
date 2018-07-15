output=[]

def decide_gradient(n):
    if(n==1):
        a=[True]
        b=[False]
        output.append(a)
        output.append(b)
        return output

    else:
        c=[True]
        d=[False]
        haha=decide_gradient(n-1)
        xixi=[]
        for i in haha:
            tmp=i+c
            xixi.append(tmp)
            tmpp=i+d
            xixi.append(tmpp)
        return xixi