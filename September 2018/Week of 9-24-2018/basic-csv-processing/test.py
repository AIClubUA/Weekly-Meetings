"""

leete ='asdf'
vartwo = '17'
varthree = 8
varfloat = 5.0 or 5.00001

firstlist = [1,2,3,4,vartwo,varfloat]
secondlist = [[0,1], [1,2]]
print(firstlist)
print('##################')

firstlist.append(vartwo)
firstlist.insert(3, 85)
firstlist.remove(4)
print(firstlist)

"""

dic1 = {
    'hello' : 5,
    'new' : 5.8,
    'string' : 'hi'
}

"""
print(dic1.keys())
print(dic1['string'])
print(dic1['new'])
"""

for k, v in dic1.items():
    print(k,v)


"""
hello 5
new 5.8
string hi
"""

if 'hi' in dic1.values() or 2==1:
    print("true")

