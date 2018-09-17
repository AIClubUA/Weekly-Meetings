
"""
Block comments
"""
# single line comments

"""
integers: any number without decimal
    2, 45, 67, 1000000

floats: any number with a decimal
    45.3333333, 28.23, 1.000000

strings: regular text
    "hello ai club", "these are called strings", "a"
"""
"""
variable names:
    nameName
    nAMENAMEname
    45name
    name_name

"""
integer_45 = 45

float_ex = 23.999

integer_45 = 45.45454545

string_1 = "here is one string"
string_2 = 'here is another string'

example = 'with this type, you can include " characters'

"""
Converting between:

    integers and floats

    int(var)
    float(var)
"""

float_1 = 45.333

# to see what data you have use: type(var)
label_1 = "Initial Float:"

print(label_1 + ' ' +str(float_1))
print(label_1, type(float_1))

# int(var) takes var to become an int
new_int = int(float_1)

print(new_int)
print(type(new_int))

new_float = float(new_int)

print(new_float)
print(type(new_float))
#-------------------------------------------------
"""
converting strings:
    str(var)
"""
print("---------")
new_int = 45
print(new_int)

string_45 = "45"
print(type(string_45))

int_from_string = int(string_45)
print(int_from_string)
print(type(int_from_string))

