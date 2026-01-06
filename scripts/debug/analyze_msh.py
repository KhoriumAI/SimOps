import sys

def analyze(path):
    print(f"Analyzing {path}")
    with open(path, 'r') as f:
        content = f.read()

    if '$MeshFormat' in content:
        header = content.split('$MeshFormat')[1].strip().split('\n')[0]
        print(f"Mesh Format: {header}")
    
    if '$Elements' in content:
        el_section = content.split('$Elements')[1].split('$EndElements')[0].strip().split('\n')
        header = el_section[0].split()
        print(f"Elements Header: {header}")
        
        # MSH 4 check
        if header[0].startswith('4') or (len(header) == 4 and '.' not in header[0]):
             # numEntityBlocks numElements minTag maxTag
             num_blocks = int(header[0])
             print(f"Num Blocks: {num_blocks}")
             
             curr = 1
             for i in range(num_blocks):
                 if curr >= len(el_section): break
                 block_header = el_section[curr].split()
                 if not block_header: continue
                 # tag(0) dim(1) type(2) num(3)
                 print(f"Block {i}: {block_header}")
                 try:
                     num = int(block_header[3])
                     curr += num + 1
                 except:
                     print("Error parsing block")
                     break
        else:
             print("Not MSH 4.x element structure?")
             
if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze(sys.argv[1])
