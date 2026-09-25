import os
import glob
import hashlib

def hash_file(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def main():
    out_md = r'E:\SleepApnea\SleepApneaSSL\experiment_reports\REPRESENTATION_ARTIFACT_AUDIT.md'
    
    # We want to check: A0, A1, A3 (multiseed), SoftCLT (softclt), Kernel Ablation (kernel_ablation)
    base_dir = r'E:\SleepApnea\SleepApneaSSL\results'
    
    all_files = glob.glob(os.path.join(base_dir, '**', '*_encoder.pth'), recursive=True)
    
    results = []
    
    for f in all_files:
        path = os.path.relpath(f, base_dir)
        h = hash_file(f)
        
        # Determine validity based on folder and timestamp/knowledge of bug fix
        # The bug fix was applied BEFORE softclt and kernel_ablation runs.
        # The multiseed folder was generated BEFORE the bug fix.
        if path.startswith('multiseed'):
            if 'seed_42' in path:
                validity = 'VALID FOR REPRESENTATION ANALYSIS (True Seed 42)'
            else:
                validity = 'INVALID / PRE-SEED-FIX (Duplicate of Seed 42)'
        elif path.startswith('softclt') or path.startswith('kernel_ablation'):
            validity = 'VALID FOR REPRESENTATION ANALYSIS'
        else:
            validity = 'UNKNOWN'
            
        results.append({
            'file': path,
            'hash': h,
            'validity': validity
        })
        
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write("# Representation Artifact Audit\n\n")
        f.write("## Checkpoint Inventory\n")
        f.write("| Path | Hash (SHA-256) | Validity |\n")
        f.write("|---|---|---|\n")
        for r in sorted(results, key=lambda x: x['file']):
            f.write(f"| {r['file']} | {r['hash'][:10]}... | {r['validity']} |\n")
            
    print(f"Audit written to {out_md}")

if __name__ == '__main__':
    main()
