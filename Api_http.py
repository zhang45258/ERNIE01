# coding:utf-8
import json
from wsgiref.simple_server import make_server
from predict_txt import main, txt2tsv, sort
mn = main()
# 定义函数，参数是函数的两个参数，都是python本身定义的，默认就行了。
def application(environ, start_response):
    # 定义文件请求的类型和当前请求成功的code
    start_response('200 OK', [('Content-Type', 'application/json')])
    # environ是当前请求的所有数据，包括Header和URL，body
    request_body = environ["wsgi.input"].read(int(environ.get("CONTENT_LENGTH", 0)))
    request_body = json.loads(request_body.decode('utf-8'))
    text = request_body["txt"]
    listData = request_body["listData"]
    print(text)
    print(listData)

    '''把text添加到mag1的每个列表后面，替换掉首次匹配的值 '''
    for i in listData:
        i[4] = text
    txt2tsv(listData=listData)  #写入文档方法
    mag_predict = mn.predict()  #进行匹配方法
    mag2 = sort(listData, mag_predict)  #排序方法
    #list包装成json
    python2json = {}
    python2json["listData"] = mag2
    data2json = json.dumps(python2json).encode('utf8')
    #返回
    return [data2json]

if __name__ == "__main__":
    port = 6089
    httpd = make_server("0.0.0.0", port, application)
    print("serving http on port {0}...".format(str(port)))
    httpd.serve_forever()

