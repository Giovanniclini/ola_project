# ola_project
A project for OLA corse at PoliMI

L'obiettivo è scegliere la combinazione migliore di prezzi per le cinque tipologie di prodotti. 
Per ogni prodotto ci sono 4 prezzi possibili (definiti da noi).
Ad ogni prezzo è associata una probabilità p_prezzo (corrisponde a quella dell'environment del prof).
Ad ogni classe di utente è associata un'altra probabilità che dovrebbe rappresentare la conversion rate, ovvero la probabilità che quel prodotto venga acquistato dato un prezzo pullato dalla probabilità precedente.

La parte di social influence determinerà come l'utente si muove dalle pagine del sito ed è quella che impone i limiti su quante pagine l'utente visita. Ad ogni visita ad una pagina, l'algoritmo di pricing si aggiorna. Ad ogni visita viene restituita la classe dell'utente, l'esito (0 se non compra, 1 se compra) su un prodotto specifico ad un prezzo specifico assime alla quantità acquistata (?).

A questo punto l'algoritmo di pricing pullerà l'arm sulla base del prodotto acquistato a quel prezzo, indicando al Learner quale tipologia di utente lo ha acquistato, così da scegliere con quale probabilità vengono aggiornate le reward e i valori necessari per l'interazione successiva. 

