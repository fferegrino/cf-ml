from unidecode import unidecode
import pandas as pd
import mananeras

# %% tags=["parameters"]
upstream = ['download-transcripts'] 
product = None
# %%

conferencias = mananeras.lee_todas(str(upstream['download-transcripts']))

# %%
dialogos_speakers = []
for conferencia in conferencias:
    for participación in conferencia.participaciones:
        participante = unidecode(participación.hablante).lower()
        label = None
        if "andres manuel" in participante:
            label = "amlo"
        elif "lopez-gatell" in participante:
            label = "lopez-gatell"
        
        if label:
            dialogos_speakers.extend(
                [(label, dialogo, len(dialogo)) for dialogo in participación.dialogos]
            )

dialogos_df = pd.DataFrame(dialogos_speakers, columns=["speaker", "dialog", "length"])
# %%

dialogos_df.to_csv(product['dialogs'], index=False)