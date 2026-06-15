# Experiment Log

## Exp-001: Baseline Configuration
- **Date**: 2026-06-15
- **Config**: r=16, alpha=32, lr=2e-4, batch=16, max_seq_length=2048
- **Diagnostic Results**:
  - Speed: 9.72s/step
  - Loss: 4.42 → 1.56 (10 steps)
  - Gradient: healthy
  - VRAM: 6.41 GB
- **Conclusion**: Speed slower than expected (vs Unsloth benchmark)
- **Next**: Investigate speed discrepancy

## Exp-002: Optimized Configuration (packing=True)
- **Date**: 2026-06-15
- **Config**: r=16, alpha=32, lr=2e-4, batch=16, max_seq_length=512, packing=True
- **Diagnostic Results**:
  - Speed: 26.87s/step (0.4x vs baseline)
  - Loss: normal
- **Conclusion**: Packing makes training slower for short sequences
- **Next**: Try without packing

## Exp-003: Reduced Sequence Length
- **Date**: 2026-06-15
- **Config**: r=16, alpha=32, lr=2e-4, batch=16, max_seq_length=512, packing=False
- **Diagnostic Results**:
  - Speed: 8.57s/step (1.1x vs baseline)
  - Loss: normal
- **Conclusion**: Minimal speedup from reducing sequence length
- **Next**: Accept baseline configuration

---

## Template

```markdown
## Exp-XXX: [Name]
- **Date**: YYYY-MM-DD
- **Config**: [key parameters]
- **Diagnostic Results**:
  - Speed: X.XXs/step
  - Loss: initial → final
  - Gradient: status
  - VRAM: X.XX GB
- **Conclusion**: [findings]
- **Next**: [next steps]
```
