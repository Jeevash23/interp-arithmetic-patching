"""Typeset the completed Markdown review draft with tables and scientific figures."""
import re
from html import escape
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,KeepTogether

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'paper/manuscript.md'
OUTPUT=HERE/'output/pdf/intermediate-value-readout-paper.pdf'
OUTPUT.parent.mkdir(parents=True,exist_ok=True)
for name,file in [('Serif','LiberationSerif-Regular'),('SerifBold','LiberationSerif-Bold'),('SerifItalic','LiberationSerif-Italic'),('SerifBoldItalic','LiberationSerif-BoldItalic')]:
    pdfmetrics.registerFont(TTFont(name,f'/usr/share/fonts/liberation/{file}.ttf'))
pdfmetrics.registerFontFamily('Serif',normal='Serif',bold='SerifBold',italic='SerifItalic',boldItalic='SerifBoldItalic')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyPaper',fontName='Serif',fontSize=10.6,leading=13.3,spaceAfter=7))
styles.add(ParagraphStyle(name='PaperTitle',fontName='SerifBold',fontSize=21,leading=24,spaceAfter=12,alignment=TA_CENTER))
styles.add(ParagraphStyle(name='PaperHeading',fontName='SerifBold',fontSize=13,leading=16,spaceBefore=11,spaceAfter=6,keepWithNext=True))
styles.add(ParagraphStyle(name='CellPaper',fontName='Serif',fontSize=8.3,leading=10))
styles.add(ParagraphStyle(name='CaptionPaper',fontName='SerifItalic',fontSize=9,leading=11,spaceAfter=9))

def inline(text):
    text=escape(text.replace('—',' - ').replace('–','-').replace('≥','>='))
    text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m:f'<link href="{m[2]}" color="#164f83">{m[1]}</link>' if m[2].startswith('http') else m[1],text)
    text=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'`([^`]+)`',r'<font size="9">\1</font>',text)
    return text

def footer(canvas,doc):
    canvas.saveState();canvas.setFont('Serif',8);canvas.setFillColor(colors.HexColor('#536071'))
    canvas.drawString(48,25,'Research review draft | 9 September 2026 | Not submitted')
    canvas.drawRightString(A4[0]-48,25,str(doc.page));canvas.restoreState()

def main():
    doc=SimpleDocTemplate(str(OUTPUT),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=43,bottomMargin=43,title='When Intermediate-Value Patching Transfers a Digit',author='Research draft')
    lines=SOURCE.read_text().splitlines();story=[];i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('# '):story.append(Paragraph(inline(line[2:]),styles['PaperTitle']));i+=1;continue
        if line.startswith('## '):story.append(Paragraph(inline(line[3:]),styles['PaperHeading']));i+=1;continue
        match=re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)',line)
        if match:
            image_path=(SOURCE.parent/match[2]).resolve();im=Image(str(image_path));scale=min(doc.width/im.imageWidth,240/im.imageHeight);im.drawWidth=im.imageWidth*scale;im.drawHeight=im.imageHeight*scale
            story.append(KeepTogether([im,Spacer(1,5),Paragraph(inline(match[1]),styles['CaptionPaper'])]));i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+',c) for c in cells):rows.append(cells)
                i+=1
            cols=len(rows[0]);widths=[doc.width/cols]*cols
            if cols in [3,5]:widths=[doc.width*.42]+[doc.width*.58/(cols-1)]*(cols-1)
            if cols==6:widths=[doc.width*x for x in [.13,.12,.29,.12,.17,.17]]
            content=[[Paragraph(('<b>'+inline(c)+'</b>') if j==0 else inline(c),styles['CellPaper']) for c in r] for j,r in enumerate(rows)]
            t=Table(content,colWidths=widths,repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef5')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#718096')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f8fa')])]))
            if t.wrap(doc.width,doc.height)[1]<doc.height-45:
                story.append(KeepTogether([t,Spacer(1,10)]))
            else:story += [t,Spacer(1,10)]
            continue
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![','- ')):
            paragraph.append(lines[i].strip());i+=1
        text=' '.join(paragraph)
        if text.startswith('- '):text='&#8226; '+inline(text[2:])
        else:text=inline(text)
        story.append(Paragraph(text,styles['BodyPaper']))
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(OUTPUT)

if __name__=='__main__':main()
