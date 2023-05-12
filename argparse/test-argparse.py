import argparse

parse = argparse.ArgumentParser(description="hello")

parse.add_argument("--a",default=5,type=int,dest='d',help="operation a") #dest 别名，后续访问名字为d
parse.add_argument("--b",default=6,type=int,help="operation b")
parse.add_argument("method",type=str,choices=['mfb','mfh']) #位置参数，必选
parse.add_argument("--verbose",action="store_true",help="print or non-print") #触发器，可选参数，一旦触发则为True，否则为False


args = parse.parse_args()
print(args)
print(args.d,args.b,args.method,args.verbose)
# python test-argparse.py --a=10 --b=3  mfh --verbose
# python test-argparse.py --a=10 --b=3  mfh
#位置参数直接输入就行，可选参数需指定