"""Deterministic synthetic workloads (small / medium / document / db_result) in ru, en, de.

Usage: python gen_workloads.py [out_dir]   (seed 20261027; writes workloads.json)
"""
import json
import random
import sys
from pathlib import Path

SEED = 20261027

FIRST = {
    "ru": ["Анна", "Иван", "Мария", "Сергей", "Ольга", "Дмитрий", "Елена", "Алексей", "Наталья", "Павел",
           "Татьяна", "Михаил", "Юлия", "Андрей", "Екатерина", "Николай", "Светлана", "Владимир"],
    "en": ["Emily", "James", "Olivia", "Michael", "Sophia", "Daniel", "Grace", "Thomas", "Hannah", "Robert",
           "Chloe", "William", "Megan", "David", "Laura", "Andrew", "Rachel", "Peter"],
    "de": ["Lena", "Jonas", "Mia", "Lukas", "Hannah", "Felix", "Sophie", "Maximilian", "Laura", "Paul",
           "Julia", "Tobias", "Katharina", "Florian", "Anna", "Stefan", "Miriam", "Matthias"],
}
LAST = {
    "ru": ["Иванова", "Петров", "Смирнова", "Кузнецов", "Соколова", "Попов", "Лебедева", "Козлов", "Новикова",
           "Морозов", "Волкова", "Зайцев", "Павлова", "Семенов", "Голубева", "Виноградов"],
    "en": ["Carter", "Whitfield", "Morrison", "Hughes", "Bennett", "Fletcher", "Harper", "Lawson", "Pearce",
           "Thornton", "Gallagher", "Mitchell", "Ashworth", "Sinclair", "Donovan", "Reeves"],
    "de": ["Müller", "Schneider", "Fischer", "Weber", "Becker", "Hoffmann", "Schäfer", "Koch", "Richter",
           "Wagner", "Bauer", "Klein", "Wolf", "Neumann", "Schwarz", "Zimmermann"],
}
# Russian last names are given in the form matching a female first name when the first name is female;
# keep it simple: pair by index parity is not grammatical, so ru names use matching-gender pairs below.
RU_PAIRS_F = [("Анна", "Иванова"), ("Мария", "Смирнова"), ("Ольга", "Соколова"), ("Елена", "Лебедева"),
              ("Наталья", "Новикова"), ("Татьяна", "Волкова"), ("Юлия", "Павлова"), ("Екатерина", "Голубева"),
              ("Светлана", "Морозова"), ("Ирина", "Кузнецова")]
RU_PAIRS_M = [("Иван", "Петров"), ("Сергей", "Кузнецов"), ("Дмитрий", "Попов"), ("Алексей", "Козлов"),
              ("Павел", "Морозов"), ("Михаил", "Зайцев"), ("Андрей", "Семенов"), ("Николай", "Виноградов"),
              ("Владимир", "Лебедев"), ("Константин", "Соколов")]
STREETS = {
    "ru": ["ул. Ленина", "Тверская улица", "пр-т Мира", "ул. Гагарина", "Невский проспект", "ул. Садовая",
           "Ломоносовский проспект", "ул. Пушкина", "Кутузовский проспект", "ул. Победы"],
    "en": ["Maple Street", "Oak Avenue", "Kingsley Road", "Elm Drive", "Harbour Lane", "Station Road",
           "Willow Court", "Park Avenue", "Victoria Street", "Bridge Road"],
    "de": ["Hauptstraße", "Lindenallee", "Bahnhofstraße", "Goethestraße", "Schillerplatz", "Bergweg",
           "Gartenstraße", "Kirchgasse", "Rosenweg", "Marktplatz"],
}
CITIES = {
    "ru": ["Москва", "Санкт-Петербург", "Казань", "Новосибирск", "Екатеринбург", "Самара"],
    "en": ["Manchester", "Leeds", "Bristol", "Springfield", "Portland", "Oxford"],
    "de": ["Berlin", "München", "Hamburg", "Köln", "Leipzig", "Stuttgart"],
}
EMAIL_DOM = ["example.com", "mail.example.org", "post.example.net"]


def name(rng, lang):
    if lang == "ru":
        return " ".join(rng.choice(rng.choice([RU_PAIRS_F, RU_PAIRS_M])))
    return f"{rng.choice(FIRST[lang])} {rng.choice(LAST[lang])}"


def phone(rng, lang):
    if lang == "ru":
        return f"+7 9{rng.randint(10,99)} {rng.randint(100,999)}-{rng.randint(10,99)}-{rng.randint(10,99)}"
    if lang == "de":
        return f"+49 1{rng.randint(51,79)} {rng.randint(1000000,9999999)}"
    return f"+44 7{rng.randint(100,999)} {rng.randint(100000,999999)}"


def address(rng, lang):
    st, city = rng.choice(STREETS[lang]), rng.choice(CITIES[lang])
    if lang == "ru":
        return f"г. {city}, {st}, д. {rng.randint(1,120)}, кв. {rng.randint(1,300)}"
    if lang == "de":
        return f"{st} {rng.randint(1,120)}, {rng.randint(10000,99999)} {city}"
    return f"{rng.randint(1,250)} {st}, {city} {rng.choice(['M1','LS2','BS1','OX4'])} {rng.randint(1,9)}AB"


TRANSLIT = str.maketrans({"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
                          "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
                          "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts",
                          "ч": "ch", "ш": "sh", "щ": "sch", "ы": "y", "ь": "", "ъ": "", "э": "e", "ю": "yu",
                          "я": "ya", "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})


def email(rng, person):
    first, last = person.lower().translate(TRANSLIT).split()[:2]
    return f"{first}.{last}{rng.randint(1,99)}@{rng.choice(EMAIL_DOM)}"


SMALL = {
    "ru": "Добрый день, это {n}, мой телефон {p}, перезвоните мне, пожалуйста, сегодня после обеда насчёт заказа.",
    "en": "Hi, this is {n}, my phone is {p}, please call me back this afternoon about the order I placed yesterday.",
    "de": "Hallo, hier ist {n}, meine Telefonnummer ist {p}, bitte rufen Sie mich heute Nachmittag wegen der Bestellung zurück.",
}

MEDIUM = {
    "ru": """Здравствуйте!

Меня зовут {n1}, я пишу по поводу заказа, который мы оформили на прошлой неделе. Заказ должен был быть доставлен по адресу {a1}, однако курьер так и не приехал, а в личном кабинете статус до сих пор показывает «передан в доставку». Я несколько раз пыталась дозвониться в службу поддержки, но каждый раз после долгого ожидания звонок обрывался.

Хочу уточнить несколько моментов. Во-первых, когда можно ожидать доставку и можно ли перенести её на выходные, так как в будние дни дома никого нет до семи вечера. Во-вторых, если доставка невозможна, прошу оформить возврат средств на ту же карту, с которой производилась оплата. В-третьих, в заказе была позиция, которую мы хотели бы заменить на аналогичную в другом цвете, если это не задержит отправку.

Мой коллега {n2} оформлял похожий заказ в вашем магазине месяц назад, и у него всё прошло без проблем, поэтому надеюсь, что это недоразумение. Если потребуется, он может подтвердить, что мы постоянные клиенты. Также можно связаться с моим мужем, {n3}, он будет дома в субботу и сможет принять посылку.

Для связи лучше всего использовать электронную почту {e1} или телефон {p1}. Если удобнее, можете написать в мессенджер на этот же номер. Пожалуйста, сообщите номер обращения, чтобы в следующий раз не объяснять ситуацию заново.

Буду благодарна за быстрый ответ. Это уже второй случай задержки за последние полгода, и мне бы очень не хотелось отказываться от услуг вашего магазина, которым мы пользуемся уже несколько лет.

С уважением,
{n1}""",
    "en": """Hello,

My name is {n1} and I am writing about the order we placed last week. It was supposed to be delivered to {a1}, but the courier never arrived and the account page still says "handed over for delivery". I tried calling your support line several times, but each time the call dropped after a long wait in the queue.

I would like to clarify a few things. First, when can we expect the delivery, and could it be moved to the weekend, since nobody is at home on weekdays before seven in the evening? Second, if delivery is not possible, please refund the payment to the same card that was used for the purchase. Third, the order contained one item we would like to swap for the same model in a different colour, provided this does not delay the shipment.

My colleague {n2} placed a similar order with your shop a month ago and everything went smoothly, so I hope this is just a misunderstanding. If needed, he can confirm that we are regular customers. You can also contact my husband, {n3}, who will be at home on Saturday and can accept the parcel.

The best way to reach me is by email at {e1} or by phone on {p1}. If it is easier, you can also send a message to the same number. Please let me know the ticket number so that I do not have to explain the situation again next time.

I would appreciate a quick reply. This is the second delay in the last six months, and I would really not like to stop using your shop, which we have relied on for several years.

Kind regards,
{n1}""",
    "de": """Guten Tag,

mein Name ist {n1} und ich schreibe Ihnen wegen der Bestellung, die wir letzte Woche aufgegeben haben. Sie sollte an die Adresse {a1} geliefert werden, doch der Kurier ist nie erschienen, und im Kundenkonto steht weiterhin „an den Versand übergeben“. Ich habe mehrmals versucht, Ihren Kundendienst telefonisch zu erreichen, aber jedes Mal brach der Anruf nach langer Wartezeit ab.

Ich möchte einige Punkte klären. Erstens: Wann können wir mit der Lieferung rechnen, und kann sie auf das Wochenende verschoben werden, da unter der Woche vor sieben Uhr abends niemand zu Hause ist? Zweitens: Falls eine Lieferung nicht möglich ist, erstatten Sie den Betrag bitte auf dieselbe Karte, mit der bezahlt wurde. Drittens enthielt die Bestellung einen Artikel, den wir gerne gegen dasselbe Modell in einer anderen Farbe tauschen würden, sofern sich der Versand dadurch nicht verzögert.

Mein Kollege {n2} hat vor einem Monat eine ähnliche Bestellung bei Ihnen aufgegeben, und alles lief reibungslos, daher hoffe ich, dass es sich um ein Missverständnis handelt. Bei Bedarf kann er bestätigen, dass wir Stammkunden sind. Sie können auch meinen Mann, {n3}, kontaktieren, der am Samstag zu Hause ist und das Paket annehmen kann.

Am besten erreichen Sie mich per E-Mail unter {e1} oder telefonisch unter {p1}. Wenn es einfacher ist, können Sie auch eine Nachricht an dieselbe Nummer schicken. Bitte teilen Sie mir die Vorgangsnummer mit, damit ich die Situation beim nächsten Mal nicht erneut schildern muss.

Ich wäre für eine schnelle Antwort dankbar. Dies ist bereits die zweite Verzögerung in den letzten sechs Monaten, und ich würde Ihren Shop, den wir seit mehreren Jahren nutzen, nur ungern verlassen.

Mit freundlichen Grüßen
{n1}""",
}

FILLER = {
    "ru": ["Стороны договорились, что все изменения к настоящему договору оформляются в письменной форме.",
           "Оплата производится ежемесячно не позднее десятого числа месяца, следующего за отчётным.",
           "Исполнитель обязуется выполнить работы качественно и в сроки, установленные календарным планом.",
           "Заказчик вправе в любое время проверять ход и качество выполняемых работ, не вмешиваясь в деятельность исполнителя.",
           "В случае просрочки исполнения обязательств виновная сторона уплачивает неустойку в размере одной десятой процента за каждый день.",
           "Отчёт за третий квартал показал рост выручки на двенадцать процентов по сравнению с прошлым годом.",
           "Основной причиной снижения маржинальности стало увеличение затрат на логистику и складское хранение.",
           "Комиссия рассмотрела представленные документы и не выявила существенных нарушений.",
           "Все споры, возникающие из настоящего договора, разрешаются путём переговоров, а при недостижении согласия — в суде.",
           "Настоящий договор вступает в силу с момента подписания и действует до полного исполнения сторонами своих обязательств.",
           "Работы по монтажу оборудования были завершены с опозданием на две недели из-за задержки поставки комплектующих.",
           "По итогам совещания было решено перенести запуск новой линии на начало следующего года.",
           "Акт приёмки подписывается в течение пяти рабочих дней с момента получения уведомления о завершении работ.",
           "Конфиденциальная информация не подлежит разглашению третьим лицам без письменного согласия другой стороны.",
           "Гарантийный срок на выполненные работы составляет двенадцать месяцев с даты подписания акта."],
    "en": ["The parties agree that any amendment to this agreement shall be made in writing and signed by both parties.",
           "Payment is due monthly no later than the tenth day of the month following the reporting period.",
           "The contractor undertakes to perform the work to a professional standard and within the agreed schedule.",
           "The client may inspect the progress and quality of the work at any time without interfering with the contractor.",
           "In the event of late performance, the defaulting party shall pay a penalty of one tenth of a percent per day.",
           "The third-quarter report showed revenue growth of twelve percent compared with the previous year.",
           "The main driver of the margin decline was the increase in logistics and warehousing costs.",
           "The committee reviewed the submitted documents and found no material irregularities.",
           "Any dispute arising from this agreement shall be resolved by negotiation and, failing that, in court.",
           "This agreement enters into force on signature and remains in effect until both parties have fulfilled their obligations.",
           "Installation of the equipment was completed two weeks late because of a delay in the delivery of components.",
           "Following the meeting it was decided to postpone the launch of the new line until early next year.",
           "The acceptance certificate shall be signed within five working days of receipt of the completion notice.",
           "Confidential information shall not be disclosed to third parties without the written consent of the other party.",
           "The warranty period for the completed work is twelve months from the date of the acceptance certificate."],
    "de": ["Die Parteien vereinbaren, dass Änderungen dieses Vertrags der Schriftform bedürfen.",
           "Die Zahlung erfolgt monatlich spätestens am zehnten Tag des auf den Abrechnungszeitraum folgenden Monats.",
           "Der Auftragnehmer verpflichtet sich, die Arbeiten fachgerecht und innerhalb des vereinbarten Zeitplans auszuführen.",
           "Der Auftraggeber darf den Fortschritt und die Qualität der Arbeiten jederzeit prüfen, ohne in die Tätigkeit des Auftragnehmers einzugreifen.",
           "Bei Verzug zahlt die säumige Partei eine Vertragsstrafe von einem Zehntel Prozent pro Tag.",
           "Der Bericht für das dritte Quartal zeigte ein Umsatzwachstum von zwölf Prozent gegenüber dem Vorjahr.",
           "Hauptursache für den Margenrückgang war der Anstieg der Logistik- und Lagerkosten.",
           "Die Kommission hat die eingereichten Unterlagen geprüft und keine wesentlichen Unregelmäßigkeiten festgestellt.",
           "Streitigkeiten aus diesem Vertrag werden einvernehmlich und, falls dies scheitert, gerichtlich beigelegt.",
           "Dieser Vertrag tritt mit Unterzeichnung in Kraft und gilt bis zur vollständigen Erfüllung der Pflichten beider Parteien.",
           "Die Montage der Anlage wurde wegen einer verspäteten Lieferung von Bauteilen mit zwei Wochen Verspätung abgeschlossen.",
           "Nach der Besprechung wurde beschlossen, den Start der neuen Linie auf Anfang nächsten Jahres zu verschieben.",
           "Das Abnahmeprotokoll ist innerhalb von fünf Werktagen nach Eingang der Fertigstellungsanzeige zu unterzeichnen.",
           "Vertrauliche Informationen dürfen Dritten ohne schriftliche Zustimmung der anderen Partei nicht offengelegt werden.",
           "Die Gewährleistungsfrist für die ausgeführten Arbeiten beträgt zwölf Monate ab dem Datum des Abnahmeprotokolls."],
}
PII_SENT = {
    "ru": ["Ответственным лицом со стороны заказчика назначен {n}.",
           "Документы были переданы {n} лично в офисе по адресу {a}.",
           "По вопросам исполнения договора следует обращаться к {n} по телефону {p}.",
           "Согласно служебной записке, {n} проживает по адресу {a} и готов принять корреспонденцию.",
           "Проверку провела {n}, замечания направлены на почту {e}."],
    "en": ["The client's designated contact person is {n}.",
           "The documents were handed to {n} in person at the office at {a}.",
           "Questions about performance of the agreement should be directed to {n} on {p}.",
           "According to the internal memo, {n} lives at {a} and can accept correspondence.",
           "The review was carried out by {n}, and the comments were sent to {e}."],
    "de": ["Ansprechpartner auf Seiten des Auftraggebers ist {n}.",
           "Die Unterlagen wurden {n} persönlich im Büro in der {a} übergeben.",
           "Fragen zur Vertragserfüllung sind an {n} unter {p} zu richten.",
           "Laut interner Notiz wohnt {n} in der {a} und kann Post entgegennehmen.",
           "Die Prüfung wurde von {n} durchgeführt, die Anmerkungen gingen an {e}."],
}
HEAD = {"ru": "Раздел", "en": "Section", "de": "Abschnitt"}
COLS = {"ru": "id | ФИО | телефон | email | адрес | сумма заказа, ₽",
        "en": "id | full_name | phone | email | address | order_total_gbp",
        "de": "id | name | telefon | email | adresse | bestellsumme_eur"}


def fill(rng, lang, tpl):
    n = name(rng, lang)
    return tpl.format(n=n, a=address(rng, lang), p=phone(rng, lang), e=email(rng, n))


def build(lang, rng):
    out = {}
    out["small"] = SMALL[lang].format(n=name(rng, lang), p=phone(rng, lang))
    n1 = name(rng, lang)
    out["medium"] = MEDIUM[lang].format(n1=n1, n2=name(rng, lang), n3=name(rng, lang), a1=address(rng, lang),
                                        e1=email(rng, n1), p1=phone(rng, lang))
    paras, words, sec = [], 0, 1
    while words < 5000:
        if len(paras) % 6 == 0:
            paras.append(f"{HEAD[lang]} {sec}.")
            sec += 1
        sents = [rng.choice(FILLER[lang]) for _ in range(rng.randint(4, 7))]
        sents.insert(rng.randint(0, len(sents)), fill(rng, lang, rng.choice(PII_SENT[lang])))
        p = " ".join(sents)
        paras.append(p)
        words += len(p.split())
    out["document"] = "\n\n".join(paras)
    rows = [COLS[lang], "-" * 40]
    for i in range(1, 501):
        n = name(rng, lang)
        rows.append(f"{i} | {n} | {phone(rng, lang)} | {email(rng, n)} | {address(rng, lang)} | "
                    f"{rng.randint(500, 250000) / (1 if lang == 'ru' else 100):.2f}")
    out["db_result"] = "\n".join(rows)
    return out


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    data = {}
    for lang in ("ru", "en", "de"):
        data[lang] = build(lang, random.Random(f"{SEED}-{lang}"))
    (out_dir / "workloads.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    for lang, d in data.items():
        print(lang, {k: (len(v), len(v.split())) for k, v in d.items()})


if __name__ == "__main__":
    main()
