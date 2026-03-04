L = input("Nhập vào list L : ")
L = L.split(",")
L = list(map(int,L))
kq = True
for i in range(len(L) - 1):
    if (L[i] + L[i + 1]) % 2 == 0:
        kq = False
        break
if kq:
    print("L là list chẵn lẻ !")
else:
    print("L không là list chẵn lẻ !")