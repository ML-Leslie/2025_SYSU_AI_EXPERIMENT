"""
这是注释
"""

# 这也是注释

"""message = "haha"
print(message)

message_2 = 2025
print(message,message_2,sep = "ME" ,end = "YOU")"""

"""print(abs(round(-5/3,2)))

name = "leslie Chan"
print(name.title())
print(name.upper())
print(name.lower())
print((name.capitalize() + " ") *3)"""

"""str = " hei  "
print(str.strip())
print(str.rstrip())
print(str.lstrip())

print(str.split())
print(str.replace("hei","haha"))

print(int(str))"""

"""a = 1
b = 200
print(int(a or b))"""

"""while(True):
    a = int(input())
    if a > 100:
        print("over 100!")
    elif 50 < a <= 100:
        print("over 50 but not more than 100!")
    else:
        print("not more than 100!")"""

"""s = 0
for i in range(100):
    s += i + 1
print(s)"""

"""bicycles = ["trek", "cannondale", "redline", "specialized"]
print(bicycles)
bicycles.append("1234")
bicycles.insert(1,"123")
print(bicycles)"""


# list = [1,2,34,21,46,33,5]
# list_1 = sorted(list)
# print(list_1)
# list_2 = sorted(list , reverse = True)
# print(list_2)
# list.sort()
# print(list)


# favorite_languages = {'jen': 'python', 'sarah': 'c','edward': 'ruby', 'phil': 'python'}
#
# for name in favorite_languages.keys():
#     print(name.title())
#
# for language in favorite_languages.values():
#     print(language.title())


# nums = [1,35,346,823,23,57,83,33]
#
# def BinarySearch(nums, target):
#     if len(nums) == 0:
#         return -1
#     elif nums.count(target) != 0:
#         return nums.index(target)
#     else:
#         return -1
#
# test = int(input())
# print(BinarySearch(nums[:],test))

# def matrixAdd(a,b,size):
#     res =[]
#     for i in range(size):
#         temp = []
#         for j in range(size):
#             temp.append(a[i][j] + b[i][j])
#         res.append(temp)
#     return res
#
# def chuck_list(lst,step):
#     return [lst[i:i+step] for i in range(0,len(lst),step)]
#
#
# def matrix2(a,b,size):
#     res = []
#     for z in range (size*size):
#         for i in range(size):
#             for j in range(size):
#                 temp = a[i][j]*b[j][i]
#         res.append(temp)
#     return chuck_list(res,size)
#
# # C = [[0] * 3 for _ in range(3)]
#
# A = []
# B = []
#
# print("输入n的值：\n")
# n = int(input())
# for i in range(n):
#     temp = []
#     for j in range(n):
#         temp.append(i*n+j+1)
#     A.append(temp)
#
# for i in range(n):
#     B.append(list(reversed(A[i])))
#
# print(matrix2(A,B,n))

# dict1 = {'Alice':'001','Bob':'002'}
#
# def ReverseKeyValue(dict):
#     dict_res = {}
#     for key,num in dict.items():
#         dict_res[num] = key
#     return dict_res
#
# print(ReverseKeyValue(dict1))


# class StuData:
#     def __init__(self, str):
#         self.data = []
#         with open(str) as file_object:
#             for line in file_object.readlines():
#                 self.data.append(line.split())
#
#     def mysort(self, str):
#         if str == "name":
#             self.data.sort(key=lambda x: x[0])
#         elif str == "num":
#             self.data.sort(key=lambda x: x[1])
#         elif str == "gender":
#             self.data.sort(key=lambda x: x[2])
#         elif str == "age":
#             self.data.sort(key=lambda x: x[3])
#
#     def exportfile(self):
#         with open("res.txt", 'w') as file_object:
#             for i in self.data:
#                 file_object.write(" ".join(i) + "\n")
#
#
#
# f = StuData("test.txt")
# f.mysort("num")
# f.exportfile()


def copy(list):
    list = list[:]
    list2 = []
    list2 = list
    return list2


list1 = ["ff", "svs", "sfs"]


if __name__ == "__main__":
    print(copy(list1))
