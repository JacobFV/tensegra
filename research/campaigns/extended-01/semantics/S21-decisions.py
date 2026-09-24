"""Prospective S21 development gates. No output/cache/model access."""
CELLS=('2x4','3x3','3x4','4x3','4x4','5x3','5x4')
def decisions(original,broad):
 for d in (original,broad):
  if set(d)!=set(CELLS) or any(type(n)is not int or not 0<=n<=512 for n in d.values()):raise ValueError('seven integer DEV512 counts required')
 old=('3x3','4x3','4x4');new=('2x4','5x3','5x4')
 heldout=broad['3x4']>=52 and broad['3x4']-original['3x4']>=26
 loss=sum(original[c]-broad[c] for c in old);retention=loss<=76;acquisition=all(broad[c]>=256 for c in new)
 return dict(heldout_gain=heldout,old_known_loss_complete=loss,old_known_retention=retention,new_motif_acquisition=acquisition,advance=heldout and retention and acquisition,automatic_extension=False)
