-----
رگرسیون لجستیک: از دودویی به چندکلاسه

شویوانگ جی یائوچن شی دانشگاه تگزاس A&M دانشگاه تگزاس A&M کالج استیشن، TX 77843 کالج استیشن، TX 77843 sji@tamu.edu ethanycx@tamu.edu

1 مقدمه

این مقدمه‌ای بر رگرسیون لجستیک چندکلاسه (LR) است که هدف آن ارائه یک معرفی کامل، مستقل و آسان‌فهم از LR چندکلاسه است. ما با مرور سریع LR دودویی شروع می‌کنیم و سپس LR دودویی را به حالت چندکلاسه تعمیم می‌دهیم. همچنین ارتباطات بین LR دودویی و LR چندکلاسه را بررسی می‌کنیم. این سند بر اساس یادداشت‌های درسی شویوانگ جی و توسط یائوچن شی در دانشگاه تگزاس A&M گردآوری شده است. این می‌تواند برای کلاس‌های سطح کارشناسی و کارشناسی ارشد استفاده شود.

2 رگرسیون لجستیک دودویی

رگرسیون لجستیک دودویی برچسب y; € {—1,+1} را برای نمونه داده شده x; با تخمین احتمال P(y|x;) و مقایسه با یک آستانه از پیش تعریف‌شده پیش‌بینی می‌کند. به یاد داشته باشید که تابع سیگموید به صورت زیر تعریف شده است:

e* 1 O(s) = = 1 ()= =a ()

که در آن s € R و @ نشان‌دهنده تابع سیگموید است. @ هر مقداری در R را به عددی در (0, 1) نگاشت می‌کند و در عین حال ترتیب هر دو عدد ورودی را حفظ می‌کند، زیرا 0(-) یک تابع افزایشی یکنواخت است.

احتمال به این صورت نمایش داده می‌شود:

0(w? x) ify=1 P = (yl@) {‘ —O(wle) ify=—t. این همچنین می‌تواند به صورت فشرده به شکل P(y|x) = O(ywr a), (2)

بیان شود، به دلیل این واقعیت که 6(—s) = 1 — 6(s). توجه داشته باشید که در حالت دودویی، ما فقط نیاز داریم یک احتمال را تخمین بزنیم، زیرا احتمالات برای +1 و -1 مجموعاً برابر با یک هستند.

3 رگرسیون لجستیک چندکلاسه

رگرسیون لجستیک دودویی فرض می‌کند که برچسب y; € {—1,+1} (i = 1,--- , N) است، در حالی که در حالت‌های چندکلاسه بیش از دو کلاس وجود دارد، یعنی y; € {1,2,---,&} (i = 1,--- ,N)، که در آن XK تعداد کلاس‌ها و N تعداد نمونه‌ها است. در این حالت، ما باید احتمال هر یک از کلاس‌های Kt را تخمین بزنیم. فرضیه در LR دودویی به این ترتیب به حالت چندکلاسه تعمیم داده می‌شود:

Ply = la; w) Ply = 2|a;w)

hy (x) = (3)

Ply = K\x;w)

یک فرض اساسی در اینجا این است که هیچ رابطه ترتیبی بین کلاس‌ها وجود ندارد. بنابراین ما به یک سیگنال خطی برای هر یک از کلاس‌های K نیاز داریم که باید به شرط x مستقل باشند. در نتیجه، در LR چندکلاسه، ما K سیگنال خطی را با ضرب داخلی بین ورودی x و K بردار وزن مستقل w,, k = 1,--- , K محاسبه می‌کنیم به صورت wile wax (4) whe

تا اینجا، تنها چیزی که برای به دست آوردن فرضیه باقی مانده است، نگاشت خروجی‌های خطی K (به عنوان یک بردار در R* ) به احتمالات K (به عنوان یک توزیع احتمالی در میان کلاس‌های ix) است.

3.1 Softmax

برای انجام چنین نگاشتی، ما تابع softmax را معرفی می‌کنیم که از تابع سیگموید تعمیم داده شده است و به صورت زیر تعریف می‌شود. با توجه به یک بردار K-بعدی v = Lee TeRK

[vr » V2, »UK | € ey

1 e

softmax(v) = ——— ) wha evk

(5)

به راحتی می‌توان تأیید کرد که softmax یک بردار در R* را به (0, 1)" نگاشت می‌کند. همه عناصر در بردار خروجی softmax مجموعاً برابر با | هستند و ترتیب آن‌ها حفظ می‌شود. بنابراین فرضیه در (3) می‌تواند به صورت زیر نوشته شود:

Ply = lax; w) ewre Ply = 2\a; 1 wy & haa) = | Py = Plesw) | eee 6) eel evn = P(y=K\ax;w) evKe

ما ارتباط بین تابع softmax و تابع سیگموید را با نشان دادن اینکه سیگموید در LR دودویی معادل softmax در LR چندکلاسه است زمانی که kK = 2 در بخش|4] بررسی خواهیم کرد.

3.2 Cross Entropy

ما LR چندکلاسه را با به حداقل رساندن یک تابع زیان (هزینه) بهینه‌سازی می‌کنیم که خطا بین پیش‌بینی‌ها و برچسب‌های واقعی را اندازه‌گیری می‌کند، همانطور که در LR دودویی انجام دادیم. بنابراین، ما cross-entropy را در معادله (7) معرفی می‌کنیم تا فاصله بین دو توزیع احتمالی را اندازه‌گیری کنیم.

Cross entropy به صورت زیر تعریف می‌شود:

K H(P,Q) =— 5 _ pj log(qi), (7) i=1 where P = (p,,--- ,px) and Q = (q,--: , qx) are two probability distributions. In multi-class

LR, the two probability distributions are the true distribution and predicted vector in Equation (3), respectively.

Here the true distribution refers to the one-hot encoding of the label. For label & (k is the correct class), the one-hot encoding is defined as a vector whose element being 1 at index k, and 0 everywhere else.

3.3. Loss Function

Now the loss for a training sample x in class c is given by

loss(x,y;w) = H(y,9) (8) =—)o yp log Ge (9) k — log J (10) = “ (11)

Sk,

where y denotes the one-hot vector and @ is the predicted distribution h(a;). And the loss on all samples (X;, Y;)%_, is

N K ewe x; loss(X,¥;w) =— >> Ty; =k) log (12) i=1 k=1 Se 1€ ewe es

4 Properties of Multi-class LR

4.1 Shift-invariance in Parameters

The softmax function in multi-class LR has an invariance property when shifting the parameters. Given the weights w = (wi,--- , wx), suppose we subtract the same vector wu from each of the K weight vectors, the outputs of softmax function will remain the same.

Proof. To prove this, let us denote w’ = {w/}4<,, where w/ = w; — u. We have

e(Wr-u)

PUY Be) SR ow (13) we ,-u"a ~ yw ew foul a (14) Pa —uUuU zx _ eke (15) (Sih, ew! t)e-ute (w,)” & e = SK Tate (16) Vint e(wi)e = P(y=k\x;w), (17)

which completes the proof.

4.2 Equivalence to Sigmoid

Once we have proved the shift-invariance, we are able to show that when KK = 2, the softmax-based multi-class LR is equivalent to the sigmoid-based binary LR. In particular, the hypothesis of both LR are equivalent.

Proof.

1 ewe hw (x) = cule 4 gute | ewe (18) 1 e(wi-wi) 7 @ _ e(wi-wi)T x 4 e(we—wi)Te amis (19) re W2- Ww Toa = poe | (20) Tpe(wa—wi)Te —1__ = | are” (21) eats h(a) _ lte-wele _ wD = f = r| —_— F we) o) (22)

where w = w 1 — w2. This completes the proof.

4.3 Relations between binary and multi-class LR

In the assignment, we’ve already proved that minimizing the logistic regression loss is equivalent to minimizing the cross-entropy loss with binary outcomes. We hereby show the proof again as below.

Proof.

N 1 arg min Ein (w) = arg min y » In(1 + e~ Yr" ®)

n=1 N 1 1 =arg min In dX O(Yn Ww? Ln) a 1 =argmin sy 2m Plys|@.) N 1 1 1 = in — Tlym, = +1)1 + [Yn = —1] ln —~—_~ aren yy 24 tle = +n Bae) Tle = Bae) a in ri 11 — +I 11 | = arg min — m = +1) in in n ote NY han) 1 — h(a) 1 1

= arg minplog — + (1 — p) log w qd

= arg min H({p, 1 —p},{¢,1—q})

where p = I[yp, = +1] and q = h(a,,). This completes the proof.

The equivalence between logistic regression loss and the cross-entropy loss, as proved above, shows that we always obtain identical weights w by minimizing the two losses. The equivalence between the losses, together with the equivalence between sigmoid and softmax, leads to the conclusion that the binary logistic regression is a particular case of multi-class logistic regression when KK = 2.

5 Derivative of multi-class LR

To optimize the multi-class LR by gradient descent, we now derive the derivative of softmax and cross entropy. The derivative of the loss function can thus be obtained by the chain rule.

5.1 Derivative of softmax Let p; denotes the i-th element of softmax(a@). Then for 7 = 7, we have

dp, Op, OTK Di Di Dea CF

0a; Oa; Oa; ( ) ai kK a ay _£ Deke aise (24) Qenai e**)? evi an etk — e% =; . - (25) doen e* yeni =pi(1 — pi) (26) =pi(1 — p;) (27) And for 7 £ 4, Op; Osa _ = 28 0a; 0a; ( ) 0 — e*e%i = (29) K (dopa e**)? ev ev = . (30) wie Dp et =— PiP; (31) If we unify the two cases with the Kronecker delta, we will have Opi da; —_ pildiy _ Py); where lifi=j bi = pe . OifiFxs

5.2 Derivative of cross entropy loss with softmax

The Cross Entropy Loss is given by: L=—S yjlog(pi)

ay

where p; = softmax;(a) = aes and y; denotes the 7-th element of the one-hot vector. The derivative of cross entropy is a OL J log(pi) Dax = » Yi Da, (32) Olog(p;) Op; _ S- Yi (pi) ; (33) F Op; Oar 1 Op; =- yu > (34) pi Oak 1 =— S > yi=- vil(Oni — Pr) (35) F Pi =— yn(1— pr) + >> yire (36) it¢k K =pr S_ Yi — Ye (37) i=1 =Pk — Yk (38)

Note that here we use the fact that ye yi= il.

Acknowledgements

This work was supported in part by National Science Foundation grants IIS-1908220, IIS-1908198, IIS-1908166, DBI-1147134, DBI-1922969, DBI-1661289, CHE-1738305, National Institutes of Health grant 1R21NS102828, and Defense Advanced Research Projects Agency grant N66001-172 -4031.
-----
