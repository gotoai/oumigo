"""Pressure-test prompt set — 20 questions × 10 languages.

A fixed corpus of general-purpose LLM prompts for load/stress testing oumigo
worker nodes (vLLM). The mix spans reasoning, coding, math, creative, explanation,
and long- vs short-form generation so it exercises different output lengths and
decode paths. The same 20 questions are provided in ten languages to also exercise
multilingual tokenization/decoding.

Languages: English (original), French, German, Spanish, Portuguese, Japanese,
Chinese (Simplified), Korean, Vietnamese, Thai.

============================================================
ENGLISH (original)
============================================================
 1. Explain the difference between supervised, unsupervised, and reinforcement
    learning, with one concrete example of each.
 2. Write a Python function that returns the nth Fibonacci number using
    memoization, and explain its time and space complexity.
 3. A train leaves City A at 3:00 PM going 60 mph. Another leaves City B, 180
    miles away, at 4:00 PM going 40 mph toward A. At what time do they meet?
 4. Summarize the plot of Mary Shelley's "Frankenstein" in exactly three sentences.
 5. What are the main tradeoffs between a microservices architecture and a
    monolith? When would you choose each?
 6. Write a haiku about a thunderstorm at sea.
 7. Explain how HTTPS keeps data secure, as if to a curious 12-year-old.
 8. List five practical techniques for reducing memory usage in a large Python
    application, with a one-line reason for each.
 9. Explain in plain words to a 9-year-old why "The early bird catches the worm,
    but the second mouse gets the cheese."
10. Prove that the square root of 2 is irrational.
11. Compare and contrast photosynthesis and cellular respiration.
12. Draft a polite but firm email declining a meeting request due to a scheduling
    conflict, proposing two alternative times.
13. What is the CAP theorem, and why can't a distributed system guarantee
    consistency, availability, and partition tolerance simultaneously?
14. Write a regular expression that matches a valid IPv4 address and explain each
    part of the pattern.
15. Describe the major causes and consequences of the fall of the Western Roman
    Empire.
16. Given the array [3, 1, 4, 1, 5, 9, 2, 6], walk through how quicksort would
    sort it step by step.
17. What are the current understood benefits and risks of intermittent fasting?
18. Explain recursion using a real-world analogy, then give a simple code example
    with a base case.
19. Write a short, punchy product description for a wireless ergonomic keyboard
    aimed at software developers.
20. If you could redesign the QWERTY keyboard layout from scratch for typing
    efficiency, what principles would you follow and why?

============================================================
FRENCH / Français
============================================================
 1. Expliquez la différence entre l'apprentissage supervisé, non supervisé et par
    renforcement, avec un exemple concret de chacun.
 2. Écrivez une fonction Python qui renvoie le n-ième nombre de Fibonacci en
    utilisant la mémoïsation, et expliquez sa complexité en temps et en espace.
 3. Un train quitte la ville A à 15h00 à 60 mph. Un autre quitte la ville B, à 180
    miles de distance, à 16h00 à 40 mph en direction de A. À quelle heure se
    rencontrent-ils ?
 4. Résumez l'intrigue de « Frankenstein » de Mary Shelley en exactement trois
    phrases.
 5. Quels sont les principaux compromis entre une architecture microservices et un
    monolithe ? Quand choisiriez-vous chacune ?
 6. Écrivez un haïku sur un orage en mer.
 7. Expliquez comment HTTPS protège les données, comme à un enfant de 12 ans
    curieux.
 8. Énumérez cinq techniques pratiques pour réduire l'utilisation de la mémoire
    dans une grande application Python, avec une raison en une ligne pour chacune.
 9. Expliquez avec des mots simples à un enfant de 9 ans pourquoi « L'oiseau
    matinal attrape le ver, mais la deuxième souris a le fromage ».
10. Démontrez que la racine carrée de 2 est irrationnelle.
11. Comparez et opposez la photosynthèse et la respiration cellulaire.
12. Rédigez un e-mail poli mais ferme déclinant une demande de réunion en raison
    d'un conflit d'horaire, en proposant deux créneaux alternatifs.
13. Qu'est-ce que le théorème CAP, et pourquoi un système distribué ne peut-il pas
    garantir simultanément la cohérence, la disponibilité et la tolérance au
    partitionnement ?
14. Écrivez une expression régulière qui correspond à une adresse IPv4 valide et
    expliquez chaque partie du motif.
15. Décrivez les principales causes et conséquences de la chute de l'Empire romain
    d'Occident.
16. Étant donné le tableau [3, 1, 4, 1, 5, 9, 2, 6], montrez étape par étape
    comment le tri rapide (quicksort) le trierait.
17. Quels sont les bénéfices et les risques actuellement compris du jeûne
    intermittent ?
18. Expliquez la récursivité à l'aide d'une analogie du monde réel, puis donnez un
    exemple de code simple avec un cas de base.
19. Rédigez une description de produit courte et percutante pour un clavier
    ergonomique sans fil destiné aux développeurs de logiciels.
20. Si vous pouviez repenser la disposition du clavier QWERTY à partir de zéro pour
    l'efficacité de frappe, quels principes suivriez-vous et pourquoi ?

============================================================
GERMAN / Deutsch
============================================================
 1. Erklären Sie den Unterschied zwischen überwachtem, unüberwachtem und
    bestärkendem Lernen, mit je einem konkreten Beispiel.
 2. Schreiben Sie eine Python-Funktion, die die n-te Fibonacci-Zahl mittels
    Memoisierung zurückgibt, und erklären Sie ihre Zeit- und Speicherkomplexität.
 3. Ein Zug verlässt Stadt A um 15:00 Uhr mit 60 mph. Ein anderer verlässt Stadt
    B, 180 Meilen entfernt, um 16:00 Uhr mit 40 mph in Richtung A. Um wie viel Uhr
    treffen sie sich?
 4. Fassen Sie die Handlung von Mary Shelleys „Frankenstein" in genau drei Sätzen
    zusammen.
 5. Was sind die wichtigsten Kompromisse zwischen einer Microservices-Architektur
    und einem Monolithen? Wann würden Sie welche wählen?
 6. Schreiben Sie ein Haiku über ein Gewitter auf See.
 7. Erklären Sie, wie HTTPS Daten sicher hält, so als ob Sie es einem neugierigen
    Zwölfjährigen erklären.
 8. Nennen Sie fünf praktische Techniken zur Reduzierung des Speicherverbrauchs in
    einer großen Python-Anwendung, mit je einer kurzen Begründung.
 9. Erklären Sie mit einfachen Worten einem 9-jährigen Kind, warum „Der frühe
    Vogel fängt den Wurm, aber die zweite Maus bekommt den Käse".
10. Beweisen Sie, dass die Quadratwurzel aus 2 irrational ist.
11. Vergleichen Sie Photosynthese und Zellatmung und stellen Sie sie gegenüber.
12. Verfassen Sie eine höfliche, aber bestimmte E-Mail, die eine Besprechungs-
    anfrage wegen eines Terminkonflikts ablehnt und zwei alternative Zeiten
    vorschlägt.
13. Was ist das CAP-Theorem, und warum kann ein verteiltes System nicht
    gleichzeitig Konsistenz, Verfügbarkeit und Partitionstoleranz garantieren?
14. Schreiben Sie einen regulären Ausdruck, der eine gültige IPv4-Adresse erkennt,
    und erklären Sie jeden Teil des Musters.
15. Beschreiben Sie die wichtigsten Ursachen und Folgen des Untergangs des
    Weströmischen Reiches.
16. Zeigen Sie für das Array [3, 1, 4, 1, 5, 9, 2, 6] Schritt für Schritt, wie
    Quicksort es sortieren würde.
17. Was sind die derzeit bekannten Vorteile und Risiken des intermittierenden
    Fastens?
18. Erklären Sie Rekursion anhand einer Analogie aus der realen Welt und geben Sie
    dann ein einfaches Codebeispiel mit einem Basisfall.
19. Schreiben Sie eine kurze, prägnante Produktbeschreibung für eine kabellose
    ergonomische Tastatur für Softwareentwickler.
20. Wenn Sie das QWERTY-Tastaturlayout von Grund auf für Tippeffizienz neu
    gestalten könnten, welche Prinzipien würden Sie befolgen und warum?

============================================================
SPANISH / Español
============================================================
 1. Explica la diferencia entre el aprendizaje supervisado, no supervisado y por
    refuerzo, con un ejemplo concreto de cada uno.
 2. Escribe una función en Python que devuelva el n-ésimo número de Fibonacci
    usando memoización, y explica su complejidad temporal y espacial.
 3. Un tren sale de la ciudad A a las 3:00 p. m. a 60 mph. Otro sale de la ciudad
    B, a 180 millas de distancia, a las 4:00 p. m. a 40 mph hacia A. ¿A qué hora
    se encuentran?
 4. Resume el argumento de «Frankenstein» de Mary Shelley en exactamente tres
    oraciones.
 5. ¿Cuáles son las principales ventajas y desventajas entre una arquitectura de
    microservicios y un monolito? ¿Cuándo elegirías cada una?
 6. Escribe un haiku sobre una tormenta eléctrica en el mar.
 7. Explica cómo HTTPS mantiene los datos seguros, como si se lo explicaras a un
    niño curioso de 12 años.
 8. Enumera cinco técnicas prácticas para reducir el uso de memoria en una
    aplicación grande de Python, con una razón de una línea para cada una.
 9. Explica con palabras sencillas a un niño de 9 años por qué «El pájaro
    madrugador atrapa al gusano, pero el segundo ratón se lleva el queso».
10. Demuestra que la raíz cuadrada de 2 es irracional.
11. Compara y contrasta la fotosíntesis y la respiración celular.
12. Redacta un correo electrónico cortés pero firme que rechace una solicitud de
    reunión por un conflicto de horario, proponiendo dos horarios alternativos.
13. ¿Qué es el teorema CAP y por qué un sistema distribuido no puede garantizar
    simultáneamente la consistencia, la disponibilidad y la tolerancia a
    particiones?
14. Escribe una expresión regular que coincida con una dirección IPv4 válida y
    explica cada parte del patrón.
15. Describe las principales causas y consecuencias de la caída del Imperio romano
    de Occidente.
16. Dado el arreglo [3, 1, 4, 1, 5, 9, 2, 6], explica paso a paso cómo lo
    ordenaría quicksort.
17. ¿Cuáles son los beneficios y riesgos actualmente conocidos del ayuno
    intermitente?
18. Explica la recursión usando una analogía del mundo real, y luego da un ejemplo
    de código simple con un caso base.
19. Escribe una descripción de producto breve y contundente para un teclado
    ergonómico inalámbrico dirigido a desarrolladores de software.
20. Si pudieras rediseñar la distribución del teclado QWERTY desde cero para
    lograr eficiencia al escribir, ¿qué principios seguirías y por qué?

============================================================
PORTUGUESE / Português
============================================================
 1. Explique a diferença entre aprendizado supervisionado, não supervisionado e
    por reforço, com um exemplo concreto de cada.
 2. Escreva uma função em Python que retorne o n-ésimo número de Fibonacci usando
    memoização e explique sua complexidade de tempo e espaço.
 3. Um trem sai da cidade A às 15h a 60 mph. Outro sai da cidade B, a 180 milhas
    de distância, às 16h a 40 mph em direção a A. A que horas eles se encontram?
 4. Resuma o enredo de «Frankenstein» de Mary Shelley em exatamente três frases.
 5. Quais são os principais trade-offs entre uma arquitetura de microsserviços e
    um monólito? Quando você escolheria cada uma?
 6. Escreva um haicai sobre uma tempestade no mar.
 7. Explique como o HTTPS mantém os dados seguros, como se fosse para uma criança
    curiosa de 12 anos.
 8. Liste cinco técnicas práticas para reduzir o uso de memória em uma aplicação
    Python grande, com um motivo de uma linha para cada.
 9. Explique em palavras simples para uma criança de 9 anos por que «O pássaro
    madrugador pega a minhoca, mas o segundo rato leva o queijo».
10. Prove que a raiz quadrada de 2 é irracional.
11. Compare e contraste a fotossíntese e a respiração celular.
12. Escreva um e-mail educado, mas firme, recusando um pedido de reunião devido a
    um conflito de agenda, propondo dois horários alternativos.
13. O que é o teorema CAP e por que um sistema distribuído não pode garantir
    simultaneamente consistência, disponibilidade e tolerância a partições?
14. Escreva uma expressão regular que corresponda a um endereço IPv4 válido e
    explique cada parte do padrão.
15. Descreva as principais causas e consequências da queda do Império Romano do
    Ocidente.
16. Dado o array [3, 1, 4, 1, 5, 9, 2, 6], mostre passo a passo como o quicksort o
    ordenaria.
17. Quais são os benefícios e riscos atualmente compreendidos do jejum
    intermitente?
18. Explique recursão usando uma analogia do mundo real e, em seguida, dê um
    exemplo de código simples com um caso base.
19. Escreva uma descrição de produto curta e impactante para um teclado ergonômico
    sem fio voltado para desenvolvedores de software.
20. Se você pudesse redesenhar o layout do teclado QWERTY do zero para eficiência
    de digitação, quais princípios seguiria e por quê?

============================================================
JAPANESE / 日本語
============================================================
 1. 教師あり学習、教師なし学習、強化学習の違いを、それぞれの具体例を1つずつ挙げて
    説明してください。
 2. メモ化を使ってn番目のフィボナッチ数を返すPython関数を書き、その時間計算量と
    空間計算量を説明してください。
 3. 列車が午後3時に時速60マイルで都市Aを出発します。別の列車が180マイル離れた都市B
    を午後4時に時速40マイルでAに向かって出発します。両者は何時に出会いますか？
 4. メアリー・シェリーの『フランケンシュタイン』のあらすじを、ちょうど3文で要約して
    ください。
 5. マイクロサービスアーキテクチャとモノリスの主なトレードオフは何ですか？それぞれを
    どんなときに選びますか？
 6. 海の雷雨についての俳句を1つ詠んでください。
 7. HTTPSがどのようにデータを安全に保つのかを、好奇心旺盛な12歳の子どもに説明する
    ように説明してください。
 8. 大規模なPythonアプリケーションでメモリ使用量を削減するための実践的な手法を5つ、
    それぞれ1行の理由を添えて挙げてください。
 9. 「早起きの鳥は虫を捕まえるが、2匹目のネズミがチーズを手に入れる」の意味を、9歳の
    子どもにわかるやさしい言葉で説明してください。
10. 2の平方根が無理数であることを証明してください。
11. 光合成と細胞呼吸を比較し、対比してください。
12. スケジュールの都合で会議の依頼を断る、丁寧だが毅然としたメールを、代替の日時を
    2つ提案しながら書いてください。
13. CAP定理とは何ですか。また、分散システムが一貫性・可用性・分断耐性を同時に保証
    できないのはなぜですか？
14. 有効なIPv4アドレスに一致する正規表現を書き、そのパターンの各部分を説明して
    ください。
15. 西ローマ帝国の滅亡の主な原因と結果を説明してください。
16. 配列 [3, 1, 4, 1, 5, 9, 2, 6] について、クイックソートがどのように並べ替えるかを
    段階的に説明してください。
17. 間欠的断食について、現在理解されている利点とリスクは何ですか？
18. 再帰を現実世界のたとえを使って説明し、その後、基底ケースを含む簡単なコード例を
    示してください。
19. ソフトウェア開発者向けのワイヤレス・エルゴノミクスキーボードの、短くて印象的な
    商品説明を書いてください。
20. タイピング効率のためにQWERTYキーボード配列をゼロから再設計できるとしたら、どんな
    原則に従いますか。またその理由は？

============================================================
CHINESE (Simplified) / 简体中文
============================================================
 1. 解释监督学习、无监督学习和强化学习之间的区别，并各举一个具体例子。
 2. 编写一个使用记忆化返回第 n 个斐波那契数的 Python 函数，并说明其时间复杂度和
    空间复杂度。
 3. 一列火车下午 3:00 以 60 英里/小时的速度离开城市 A。另一列火车下午 4:00 从 180
    英里外的城市 B 以 40 英里/小时的速度朝 A 出发。它们几点相遇？
 4. 用恰好三句话概括玛丽·雪莱的《弗兰肯斯坦》的情节。
 5. 微服务架构与单体架构之间的主要权衡是什么？你会在什么情况下分别选择它们？
 6. 写一首关于海上雷暴的俳句。
 7. 像向一个好奇的 12 岁孩子解释那样，说明 HTTPS 是如何保证数据安全的。
 8. 列出在大型 Python 应用中减少内存占用的五种实用技巧，并为每一种给出一行理由。
 9. 用简单的话向一个 9 岁孩子解释为什么“早起的鸟儿有虫吃，但第二只老鼠才能吃到奶酪”。
10. 证明 2 的平方根是无理数。
11. 比较并对比光合作用与细胞呼吸。
12. 起草一封礼貌但坚定的电子邮件，因日程冲突而婉拒一个会议请求，并提出两个备选时间。
13. 什么是 CAP 定理？为什么分布式系统无法同时保证一致性、可用性和分区容错性？
14. 编写一个匹配有效 IPv4 地址的正则表达式，并解释该模式的每个部分。
15. 描述西罗马帝国灭亡的主要原因和后果。
16. 给定数组 [3, 1, 4, 1, 5, 9, 2, 6]，逐步演示快速排序如何对其进行排序。
17. 间歇性禁食目前已知的益处和风险有哪些？
18. 用一个现实世界的类比来解释递归，然后给出一个包含基线情形的简单代码示例。
19. 为面向软件开发者的无线人体工学键盘写一段简短有力的产品描述。
20. 如果可以为打字效率从零开始重新设计 QWERTY 键盘布局，你会遵循哪些原则，为什么？

============================================================
KOREAN / 한국어
============================================================
 1. 지도 학습, 비지도 학습, 강화 학습의 차이를 각각의 구체적인 예를 하나씩 들어
    설명하세요.
 2. 메모이제이션을 사용하여 n번째 피보나치 수를 반환하는 Python 함수를 작성하고,
    그 시간 복잡도와 공간 복잡도를 설명하세요.
 3. 기차가 오후 3시에 시속 60마일로 도시 A를 출발합니다. 다른 기차가 180마일 떨어진
    도시 B에서 오후 4시에 시속 40마일로 A를 향해 출발합니다. 두 기차는 몇 시에
    만나나요?
 4. 메리 셸리의 『프랑켄슈타인』의 줄거리를 정확히 세 문장으로 요약하세요.
 5. 마이크로서비스 아키텍처와 모놀리스 사이의 주요 트레이드오프는 무엇인가요?
    각각 언제 선택하겠습니까?
 6. 바다의 뇌우에 관한 하이쿠를 한 편 쓰세요.
 7. 호기심 많은 12세 아이에게 설명하듯이, HTTPS가 어떻게 데이터를 안전하게 지키는지
    설명하세요.
 8. 대규모 Python 애플리케이션에서 메모리 사용량을 줄이는 실용적인 기법 다섯 가지를
    각각 한 줄의 이유와 함께 나열하세요.
 9. "일찍 일어난 새가 벌레를 잡지만, 두 번째 쥐가 치즈를 얻는다"는 말의 뜻을 9살
    아이가 이해할 수 있는 쉬운 말로 설명하세요.
10. 2의 제곱근이 무리수임을 증명하세요.
11. 광합성과 세포 호흡을 비교하고 대조하세요.
12. 일정 충돌로 인해 회의 요청을 거절하는 정중하지만 단호한 이메일을 두 개의 대체
    시간을 제안하며 작성하세요.
13. CAP 정리란 무엇이며, 분산 시스템이 일관성, 가용성, 분할 내성을 동시에 보장할 수
    없는 이유는 무엇인가요?
14. 유효한 IPv4 주소와 일치하는 정규 표현식을 작성하고 패턴의 각 부분을 설명하세요.
15. 서로마 제국 멸망의 주요 원인과 결과를 설명하세요.
16. 배열 [3, 1, 4, 1, 5, 9, 2, 6]에 대해 퀵정렬이 어떻게 정렬하는지 단계별로
    설명하세요.
17. 간헐적 단식의 현재 알려진 이점과 위험은 무엇인가요?
18. 실생활 비유를 사용하여 재귀를 설명한 다음, 기저 사례가 있는 간단한 코드 예제를
    제시하세요.
19. 소프트웨어 개발자를 대상으로 하는 무선 인체공학 키보드에 대한 짧고 강렬한 제품
    설명을 작성하세요.
20. 타이핑 효율성을 위해 QWERTY 키보드 배열을 처음부터 다시 설계할 수 있다면, 어떤
    원칙을 따르겠으며 그 이유는 무엇인가요?

============================================================
VIETNAMESE / Tiếng Việt
============================================================
 1. Giải thích sự khác biệt giữa học có giám sát, học không giám sát và học tăng
    cường, kèm một ví dụ cụ thể cho mỗi loại.
 2. Viết một hàm Python trả về số Fibonacci thứ n bằng cách sử dụng ghi nhớ
    (memoization), và giải thích độ phức tạp về thời gian và không gian của nó.
 3. Một chuyến tàu rời thành phố A lúc 3:00 chiều với vận tốc 60 dặm/giờ. Một
    chuyến khác rời thành phố B, cách 180 dặm, lúc 4:00 chiều với vận tốc 40
    dặm/giờ hướng về A. Chúng gặp nhau lúc mấy giờ?
 4. Tóm tắt cốt truyện của «Frankenstein» của Mary Shelley trong đúng ba câu.
 5. Những đánh đổi chính giữa kiến trúc microservices và kiến trúc nguyên khối
    (monolith) là gì? Khi nào bạn sẽ chọn mỗi loại?
 6. Viết một bài haiku về cơn giông trên biển.
 7. Giải thích cách HTTPS giữ an toàn cho dữ liệu, như thể đang giải thích cho một
    đứa trẻ 12 tuổi tò mò.
 8. Liệt kê năm kỹ thuật thực tế để giảm mức sử dụng bộ nhớ trong một ứng dụng
    Python lớn, kèm một lý do ngắn gọn cho mỗi kỹ thuật.
 9. Giải thích bằng lời đơn giản cho một đứa trẻ 9 tuổi vì sao "Con chim dậy sớm
    bắt được sâu, nhưng con chuột thứ hai mới lấy được phô mai".
10. Chứng minh rằng căn bậc hai của 2 là số vô tỉ.
11. So sánh và đối chiếu quang hợp với hô hấp tế bào.
12. Soạn một email lịch sự nhưng dứt khoát để từ chối một lời mời họp do trùng
    lịch, đề xuất hai khung giờ thay thế.
13. Định lý CAP là gì, và tại sao một hệ thống phân tán không thể đồng thời đảm bảo
    tính nhất quán, tính sẵn sàng và khả năng chịu phân vùng?
14. Viết một biểu thức chính quy khớp với một địa chỉ IPv4 hợp lệ và giải thích
    từng phần của mẫu.
15. Mô tả các nguyên nhân và hậu quả chính của sự sụp đổ của Đế chế La Mã phương
    Tây.
16. Với mảng [3, 1, 4, 1, 5, 9, 2, 6], hãy trình bày từng bước cách quicksort sắp
    xếp nó.
17. Những lợi ích và rủi ro hiện được hiểu biết của việc nhịn ăn gián đoạn là gì?
18. Giải thích đệ quy bằng một phép loại suy trong thế giới thực, sau đó đưa ra một
    ví dụ mã đơn giản có trường hợp cơ sở.
19. Viết một mô tả sản phẩm ngắn gọn và ấn tượng cho một bàn phím công thái học
    không dây dành cho các nhà phát triển phần mềm.
20. Nếu có thể thiết kế lại bố cục bàn phím QWERTY từ đầu để tối ưu hiệu quả gõ
    phím, bạn sẽ tuân theo những nguyên tắc nào và tại sao?

============================================================
THAI / ภาษาไทย
============================================================
 1. อธิบายความแตกต่างระหว่างการเรียนรู้แบบมีผู้สอน แบบไม่มีผู้สอน และแบบเสริมกำลัง
    พร้อมยกตัวอย่างที่เป็นรูปธรรมอย่างละหนึ่งตัวอย่าง
 2. เขียนฟังก์ชัน Python ที่คืนค่าเลขฟีโบนัชชีตัวที่ n โดยใช้การจำ (memoization)
    และอธิบายความซับซ้อนด้านเวลาและพื้นที่
 3. รถไฟออกจากเมือง A เวลา 15:00 น. ด้วยความเร็ว 60 ไมล์/ชม. อีกขบวนออกจากเมือง B
    ซึ่งอยู่ห่าง 180 ไมล์ เวลา 16:00 น. ด้วยความเร็ว 40 ไมล์/ชม. มุ่งหน้าไปยัง A
    ทั้งสองขบวนจะพบกันเวลาใด
 4. สรุปเนื้อเรื่องของ «แฟรงเกนสไตน์» ของแมรี เชลลีย์ ในสามประโยคพอดี
 5. ข้อแลกเปลี่ยนหลักระหว่างสถาปัตยกรรมไมโครเซอร์วิสกับแบบโมโนลิธคืออะไร และคุณจะ
    เลือกแต่ละแบบเมื่อใด
 6. เขียนไฮกุเกี่ยวกับพายุฝนฟ้าคะนองกลางทะเล
 7. อธิบายว่า HTTPS ปกป้องข้อมูลให้ปลอดภัยอย่างไร ราวกับกำลังอธิบายให้เด็กอายุ 12 ปี
    ที่อยากรู้อยากเห็นฟัง
 8. ระบุเทคนิคที่ใช้ได้จริงห้าอย่างในการลดการใช้หน่วยความจำในแอปพลิเคชัน Python
    ขนาดใหญ่ พร้อมเหตุผลสั้น ๆ หนึ่งบรรทัดสำหรับแต่ละข้อ
 9. อธิบายด้วยคำง่าย ๆ ให้เด็กอายุ 9 ขวบฟังว่าทำไม "นกที่ตื่นเช้าจับหนอนได้ แต่หนู
    ตัวที่สองได้กินเนยแข็ง"
10. จงพิสูจน์ว่ารากที่สองของ 2 เป็นจำนวนอตรรกยะ
11. เปรียบเทียบและชี้ความแตกต่างระหว่างการสังเคราะห์ด้วยแสงกับการหายใจระดับเซลล์
12. ร่างอีเมลที่สุภาพแต่หนักแน่นเพื่อปฏิเสธคำขอประชุมเนื่องจากตารางเวลาชนกัน พร้อม
    เสนอช่วงเวลาทางเลือกสองช่วง
13. ทฤษฎีบท CAP คืออะไร และเหตุใดระบบแบบกระจายจึงไม่สามารถรับประกันความสอดคล้อง
    ความพร้อมใช้งาน และความทนต่อการแบ่งพาร์ทิชันได้พร้อมกัน
14. เขียนนิพจน์ปรกติ (regular expression) ที่จับคู่กับที่อยู่ IPv4 ที่ถูกต้อง และ
    อธิบายแต่ละส่วนของรูปแบบ
15. อธิบายสาเหตุและผลกระทบสำคัญของการล่มสลายของจักรวรรดิโรมันตะวันตก
16. จากอาร์เรย์ [3, 1, 4, 1, 5, 9, 2, 6] จงแสดงทีละขั้นตอนว่าควิกซอร์ต (quicksort)
    จะเรียงลำดับอย่างไร
17. ประโยชน์และความเสี่ยงของการอดอาหารเป็นช่วง (intermittent fasting) ที่เข้าใจกัน
    ในปัจจุบันมีอะไรบ้าง
18. อธิบายการเรียกซ้ำ (recursion) โดยใช้การเปรียบเทียบกับสิ่งในโลกจริง จากนั้นยก
    ตัวอย่างโค้ดง่าย ๆ ที่มีกรณีฐาน
19. เขียนคำอธิบายผลิตภัณฑ์สั้น ๆ ที่โดนใจ สำหรับคีย์บอร์ดตามหลักสรีรศาสตร์แบบไร้สาย
    ที่มุ่งเป้าไปที่นักพัฒนาซอฟต์แวร์
20. หากคุณสามารถออกแบบผังคีย์บอร์ด QWERTY ใหม่ตั้งแต่ต้นเพื่อประสิทธิภาพการพิมพ์
    คุณจะยึดหลักการใดและเพราะเหตุใด
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass

import httpx

_QUESTION_RE = re.compile(r"^\s*(\d+)\.\s+(.*)$")


def parse_questions(doc: str | None) -> list[str]:
    """Extract the numbered prompts from this module's docstring (all languages).

    Each numbered item — with its wrapped continuation lines rejoined — becomes one
    prompt; separators, language headers, and the intro paragraph are ignored. The
    result is a flat, multilingual pool clients draw from at random.
    """
    questions: list[str] = []
    current: str | None = None
    for line in (doc or "").splitlines():
        stripped = line.strip()
        if stripped and set(stripped) == {"="}:            # ==== separator
            if current is not None:
                questions.append(current)
                current = None
            continue
        m = _QUESTION_RE.match(line)
        if m:
            if current is not None:
                questions.append(current)
            current = m.group(2).strip()
        elif current is not None:
            if not stripped:                               # blank line ends the item
                questions.append(current)
                current = None
            elif line[:1] in (" ", "\t"):                  # indented continuation
                current += " " + stripped
            else:                                          # a header line ends the item
                questions.append(current)
                current = None
    if current is not None:
        questions.append(current)
    return questions


@dataclass
class Record:
    """The outcome of one request — the unit the summary aggregates over."""

    client: int
    index: int
    ok: bool
    latency_s: float
    prompt_tokens: int = 0        # input tokens (prompt + accumulated history)
    completion_tokens: int = 0    # output tokens (generated)
    error: str = ""
    ttft_s: float | None = None   # time to first token (streaming only)


async def _detect_model(base_url: str, headers: dict[str, str]) -> str:
    """Ask the server which model it serves (homogeneous fleet → the first is it)."""
    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.get(f"{base_url}/v1/models", headers=headers)
        r.raise_for_status()
        return r.json()["data"][0]["id"]


def _brief(text: str, limit: int = 220) -> str:
    """Pull the human message out of an error body (vLLM returns JSON), truncated."""
    try:
        msg = json.loads(text).get("message")
        if msg:
            text = msg
    except (ValueError, TypeError, AttributeError):
        pass
    return " ".join(text.split())[:limit]


async def _one_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    headers: dict[str, str],
    stream: bool,
) -> tuple[bool, float, float | None, int, int, str, str]:
    """One chat completion → (ok, latency, ttft, prompt_tokens, completion_tokens, answer, error).

    Streaming reads SSE deltas so it can time the *first* token (TTFT); non-streaming
    reads the whole JSON at once (TTFT is None). `stream_options.include_usage` asks
    vLLM for a final usage chunk so both token counts survive streaming.
    """
    t0 = time.perf_counter()
    try:
        if stream:
            body = {**payload, "stream": True, "stream_options": {"include_usage": True}}
            parts: list[str] = []
            first: float | None = None
            prompt_tokens = completion_tokens = 0
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code >= 400:
                    detail = (await resp.aread()).decode("utf-8", "replace")
                    return (False, time.perf_counter() - t0, None, 0, 0, "",
                            f"HTTP {resp.status_code}: {_brief(detail)}")
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    usage = chunk.get("usage")
                    if usage:
                        prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                        completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
                    choices = chunk.get("choices") or []
                    if choices:
                        piece = (choices[0].get("delta") or {}).get("content")
                        if piece:
                            if first is None:
                                first = time.perf_counter()
                            parts.append(piece)
            dt = time.perf_counter() - t0
            ttft = None if first is None else first - t0
            return True, dt, ttft, prompt_tokens, completion_tokens, "".join(parts), ""
        resp = await client.post(url, json={**payload, "stream": False}, headers=headers)
        if resp.status_code >= 400:
            return (False, time.perf_counter() - t0, None, 0, 0, "",
                    f"HTTP {resp.status_code}: {_brief(resp.text)}")
        data = resp.json()
        dt = time.perf_counter() - t0
        answer = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        return True, dt, None, prompt_tokens, completion_tokens, answer, ""
    except Exception as exc:  # noqa: BLE001 - a failed call is a data point, not a crash
        return False, time.perf_counter() - t0, None, 0, 0, "", f"{type(exc).__name__}: {exc}"


_MSG_OVERHEAD = 4  # rough per-message chat-template token cost


def _estimate_tokens(text: str) -> int:
    """Tokenizer-free, script-aware token estimate — deliberately errs high.

    ASCII text averages ~4 chars/token; CJK/Thai/etc. run closer to ~1 token/char,
    so wide chars are counted ~1:1. Over-estimating means we trim history slightly
    more than strictly needed — the safe direction for staying under the limit.
    """
    ascii_n = sum(1 for c in text if c.isascii())
    return math.ceil(ascii_n / 4) + (len(text) - ascii_n)


def _fit_history(hist: list[dict], question: str, max_tokens: int, max_ctx: int) -> list[dict]:
    """Drop oldest (user, assistant) pairs until history+question+output fit `max_ctx`."""
    budget = max_ctx - max_tokens - _estimate_tokens(question) - _MSG_OVERHEAD
    if budget <= 0:
        return []
    trimmed = list(hist)
    used = sum(_estimate_tokens(m["content"]) + _MSG_OVERHEAD for m in trimmed)
    while trimmed and used > budget:
        used -= sum(_estimate_tokens(m["content"]) + _MSG_OVERHEAD for m in trimmed[:2])
        del trimmed[:2]  # oldest (user, assistant) pair
    return trimmed


async def _run_client(
    cid: int,
    base_url: str,
    model: str,
    pool: list[str],
    n_questions: int,
    history: int,
    max_tokens: int,
    headers: dict[str, str],
    out: list[Record],
    stream: bool,
    duration: float | None,
    max_context_length: int,
    verbose: bool,
) -> None:
    """One simulated client: a single multi-turn conversation.

    Runs a fixed `n_questions` turns, or — when `duration` is set — keeps asking
    until that many seconds elapse. It keeps the last `history` (user, assistant)
    pairs and prepends them to each new question, growing a realistic context.
    """
    convo: list[dict] = []  # trimmed to the last `history` (user, assistant) pairs
    url = f"{base_url}/v1/chat/completions"
    deadline = None if duration is None else time.perf_counter() + duration
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        i = 0
        while (deadline is None and i < n_questions) or (
            deadline is not None and time.perf_counter() < deadline
        ):
            question = random.choice(pool)
            hist = convo[-2 * history :] if history > 0 else []
            hist = _fit_history(hist, question, max_tokens, max_context_length)
            messages = hist + [{"role": "user", "content": question}]
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
            }
            ok, dt, ttft, ptok, ctok, answer, err = await _one_request(
                client, url, payload, headers, stream
            )
            rec = Record(cid, i, ok, dt, prompt_tokens=ptok, completion_tokens=ctok,
                         error=err, ttft_s=ttft)
            if ok and history > 0:
                convo += [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
                convo = convo[-2 * history :]
            out.append(rec)
            label = f"Q{i + 1}/{n_questions}" if deadline is None else f"Q{i + 1}"
            io = f"in {ptok:>5} out {ctok:>5} tok"
            if not ok:
                status = f"FAIL {dt:6.2f}s  {err[:56]}"
            elif ttft is not None:
                status = f"ok   {dt:6.2f}s  ttft {ttft:5.2f}s  {io}"
            else:
                status = f"ok   {dt:6.2f}s  {io}"
            print(f"[client {cid:>2}] {label}  {status}")
            if verbose:
                # The "final prompt" = the current question only (history excluded).
                print(f"    prompt   : {question}")
                if ok:
                    print(f"    response : {answer}")
            i += 1


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def _summarize(records: list[Record], wall_s: float, clients: int) -> int:
    ok = [r for r in records if r.ok]
    fails = [r for r in records if not r.ok]
    lat = [r.latency_s for r in ok]
    prompt_tok = sum(r.prompt_tokens for r in ok)
    completion_tok = sum(r.completion_tokens for r in ok)
    print("\n" + "=" * 60)
    print("PRESSURE TEST SUMMARY")
    print("=" * 60)
    print(f"  clients (concurrency) : {clients}")
    avg_s = sum(r.latency_s for r in records) / len(records) if records else 0.0
    print(f"  requests total        : {len(records)}  "
          f"(OK={len(ok)}, Err={len(fails)}, {avg_s:.2f} sec/req)")
    print(f"  wall time             : {wall_s:.2f}s")
    if records and wall_s:
        print(f"  throughput            : {len(records) / wall_s:.2f} req/s")
    if lat:
        print(f"  latency mean          : {sum(lat) / len(lat):.2f}s")
        print(f"  latency p50/p95/p99   : {_pct(lat, 0.5):.2f} / {_pct(lat, 0.95):.2f} / {_pct(lat, 0.99):.2f}s")
        print(f"  latency min/max       : {min(lat):.2f}s / {max(lat):.2f}s")
    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    if ttfts:
        print(f"  TTFT p50/p95/p99      : {_pct(ttfts, 0.5):.2f} / {_pct(ttfts, 0.95):.2f} / {_pct(ttfts, 0.99):.2f}s")
    if wall_s and (prompt_tok or completion_tok):
        total_tok = prompt_tok + completion_tok
        print(f"  prompt tokens (in)    : {prompt_tok}  ({prompt_tok / wall_s:.1f} tok/s)")
        print(f"  completion tok (out)  : {completion_tok}  ({completion_tok / wall_s:.1f} tok/s)")
        print(f"  total tokens          : {total_tok}  ({total_tok / wall_s:.1f} tok/s)")
    if fails:
        counts: dict[str, int] = {}
        for r in fails:
            counts[r.error] = counts.get(r.error, 0) + 1
        print("  errors:")
        for err, n in list(counts.items())[:5]:
            print(f"    {n:>4}x  {err}")
    print("=" * 60)
    return 0 if ok else 1


async def _main_async(args: argparse.Namespace) -> int:
    base_url = f"http://{args.host}:{args.port}"
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}

    pool = parse_questions(__doc__)
    if not pool:
        print("error: no questions found in the module docstring", file=sys.stderr)
        return 2

    model = args.model
    if not model:
        try:
            model = await _detect_model(base_url, headers)
        except Exception as exc:  # noqa: BLE001 - report and ask the user to pass --model
            print(
                f"error: could not auto-detect a model from {base_url}/v1/models ({exc}); "
                f"pass --model=<name>",
                file=sys.stderr,
            )
            return 2

    print(f"target   : {base_url}  (model={model})")
    print(f"prompts  : {len(pool)} in pool (20 questions x 10 languages)")
    if args.duration is None:
        total = args.clients * args.questions
        plan = f"{args.clients} client(s) x {args.questions} question(s) = {total} requests"
    else:
        plan = f"{args.clients} client(s) for {args.duration:.0f}s each"
    print(
        f"plan     : {plan} | history={args.history} turns | "
        f"stream={'on' if args.stream else 'off'} | max_tokens={args.max_tokens} | "
        f"ctx<={args.max_context_length}\n"
    )

    records: list[Record] = []
    t0 = time.perf_counter()
    await asyncio.gather(
        *[
            _run_client(
                cid, base_url, model, pool, args.questions, args.history,
                args.max_tokens, headers, records, args.stream, args.duration,
                args.max_context_length, args.verbose,
            )
            for cid in range(args.clients)
        ]
    )
    return _summarize(records, time.perf_counter() - t0, args.clients)


def _bool(value: str) -> bool:
    v = value.strip().lower()
    if v in ("yes", "y", "true", "t", "1", "on"):
        return True
    if v in ("no", "n", "false", "f", "0", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected yes/no, got {value!r}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Concurrent pressure test for an oumigo worker (OpenAI-compatible vLLM API).",
    )
    p.add_argument("--host", default="localhost", help="server host (default: localhost)")
    p.add_argument("--port", type=int, default=7012,
                   help="data-plane (router) port (default: 7012)")
    p.add_argument("--clients", type=int, default=1,
                   help="parallel clients = concurrency (default: 1)")
    p.add_argument("--history", type=int, default=3,
                   help="prior (Q,A) turns each client keeps as context (default: 3)")
    p.add_argument("--questions", type=int, default=5,
                   help="questions each client asks (default: 5)")
    p.add_argument("--stream", type=_bool, default=True, metavar="YES|NO",
                   help="stream responses and measure TTFT (default: YES)")
    p.add_argument("--verbose", type=_bool, default=False, metavar="YES|NO",
                   help="print each request's final prompt (no history) and response (default: NO)")
    p.add_argument("--duration", type=float, default=None, metavar="SECONDS",
                   help="run each client for N seconds instead of a fixed --questions count")
    p.add_argument("--model", default=None,
                   help="model id (default: auto-detect from /v1/models)")
    p.add_argument("--max-tokens", type=int, default=4000,
                   help="max_tokens per response (default: 4000)")
    p.add_argument("--max-context-length", type=int, default=8192,
                   help="model context window; history is trimmed so prompt+max_tokens "
                        "fit within it (default: 8192)")
    p.add_argument("--api-key", default=None,
                   help="bearer token, if the endpoint requires one")
    args = p.parse_args()
    try:
        raise SystemExit(asyncio.run(_main_async(args)))
    except KeyboardInterrupt:
        print("\ninterrupted.", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
