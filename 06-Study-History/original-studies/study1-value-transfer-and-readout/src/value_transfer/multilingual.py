"""Exploratory, [redacted]-reviewed translations; independent human review is pending."""
from .tasks import prompt as english_prompt

FORMAT_ES='Devuelve exactamente dos líneas, con solo un número entero después de cada etiqueta:\nIntermediate: <resultado del primer paso>\nFinal: <resultado del segundo paso>'
FORMAT_TA='சரியாக இரண்டு வரிகளை மட்டும் எழுதவும். ஒவ்வொரு குறிச்சொல்லுக்கும் பிறகு ஒரு முழு எண்ணை மட்டும் எழுதவும்:\nIntermediate: <முதல் படியின் முடிவு>\nFinal: <இரண்டாம் படியின் முடிவு>'

def translated_prompt(t,family,language,kind='full',supplied=None):
    if language=='en':return english_prompt(t,family,kind,supplied)
    if family not in ['dev_box','dev_score']:
        raise ValueError('Only development translations are prepared; freeze before held-out translation work.')
    a,b,k=t['a'],t['b'],t['k']
    if language=='es':
        if family=='dev_box':
            first=f'Una caja contiene inicialmente {a} fichas. '+(f'Añade {b} fichas a la caja.' if t['first']=='add' else f'Retira {b} fichas de la caja.')
        else:
            first=f'Tu puntuación inicial es de {a} puntos. '+(f'Primero ganas {b} puntos.' if t['first']=='add' else f'Primero pierdes {b} puntos.')
        remaining={'add':f'Después, suma {k} al resultado intermedio.','sub':f'Después, resta {k} al resultado intermedio.','mul':f'Después, multiplica el resultado intermedio por {k}.'}[t['remaining']]
        if kind=='full':return first+' Con esto termina el primer paso. '+remaining+' ¿Cuál es el resultado final?\n'+FORMAT_ES
        if kind=='first':return first+' ¿Cuál es el resultado de este paso?\nDevuelve exactamente una línea: Final: <número entero>'
        z=t['z'] if supplied is None else supplied
        return f'El resultado intermedio es {z}. '+remaining+' ¿Cuál es el resultado final?\nDevuelve exactamente una línea: Final: <número entero>'
    if language=='ta':
        # Stones stand for physical counting pieces; this is a task adaptation.
        if family=='dev_box':
            first=f'ஒரு பெட்டியில் முதலில் {a} கற்கள் உள்ளன. '+(f'பெட்டியில் மேலும் {b} கற்களைச் சேர்க்கவும்.' if t['first']=='add' else f'பெட்டியிலிருந்து {b} கற்களை அகற்றவும்.')
        else:
            first=f'உங்கள் ஆரம்ப மதிப்பெண் {a}. '+(f'முதலில் {b} புள்ளிகளைப் பெறுகிறீர்கள்.' if t['first']=='add' else f'முதலில் {b} புள்ளிகளை இழக்கிறீர்கள்.')
        remaining={'add':f'பின்னர் இடைநிலை முடிவுடன் {k}-ஐக் கூட்டவும்.','sub':f'பின்னர் இடைநிலை முடிவிலிருந்து {k}-ஐக் கழிக்கவும்.','mul':f'பின்னர் இடைநிலை முடிவை {k}-ஆல் பெருக்கவும்.'}[t['remaining']]
        if kind=='full':return first+' இத்துடன் முதல் படி முடிகிறது. '+remaining+' இறுதி முடிவு என்ன?\n'+FORMAT_TA
        if kind=='first':return first+' இந்தப் படியின் முடிவு என்ன?\nஒரே ஒரு வரியில் எழுதவும்: Final: <முழு எண்>'
        z=t['z'] if supplied is None else supplied
        return f'இடைநிலை முடிவு {z}. '+remaining+' இறுதி முடிவு என்ன?\nஒரே ஒரு வரியில் எழுதவும்: Final: <முழு எண்>'
    raise ValueError(language)
