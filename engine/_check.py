c = open('app.py','r',encoding='utf-8').read()
t = "    result['asset'] = asset\n    st.session_state.data = result"
print('LF found:', t in c)
t2 = t.replace('\n','\r\n')
print('CRLF found:', t2 in c)
# Check raw bytes
raw = open('app.py','rb').read()
print('Has CRLF:', b'\r\n' in raw)
print('Has bare LF:', b'\n' in raw.replace(b'\r\n', b''))
