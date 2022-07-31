import preprocessing as x

output=x.predict('ex-con back behind bar')
if output==0:
    print("not Sarcasm")
else:
    print("sarcasm")
