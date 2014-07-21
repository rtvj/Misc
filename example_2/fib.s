	.file	"fib.c"
	.globl	idx
	.bss
	.align 4
	.type	idx, @object
	.size	idx, 4
idx:
	.zero	4
	.globl	jdx
	.data
	.align 4
	.type	jdx, @object
	.size	jdx, 4
jdx:
	.long	1
	.globl	fib
	.bss
	.align 4
	.type	fib, @object
	.size	fib, 4
fib:
	.zero	4
	.globl	fib0
	.align 4
	.type	fib0, @object
	.size	fib0, 4
fib0:
	.zero	4
	.globl	fib1
	.data
	.align 4
	.type	fib1, @object
	.size	fib1, 4
fib1:
	.long	1
	.globl	seqIter
	.align 4
	.type	seqIter, @object
	.size	seqIter, 4
seqIter:
	.long	47
	.globl	loopIter
	.align 4
	.type	loopIter, @object
	.size	loopIter, 4
loopIter:
	.long	1
	.text
	.globl	wrapper
	.type	wrapper, @function
wrapper:
.LFB0:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	$0, idx(%rip)
	jmp	.L2
.L5:
	movl	fib0(%rip), %edx
	movl	fib1(%rip), %eax
	addl	%edx, %eax
	movl	%eax, fib(%rip)
	jmp	.L3
.L4:
	movl	fib1(%rip), %eax
	movl	%eax, fib0(%rip)
	movl	fib(%rip), %eax
	movl	%eax, fib1(%rip)
	movl	fib0(%rip), %edx
	movl	fib1(%rip), %eax
	addl	%edx, %eax
	movl	%eax, fib(%rip)
	movl	jdx(%rip), %eax
	addl	$1, %eax
	movl	%eax, jdx(%rip)
.L3:
	movl	jdx(%rip), %edx
	movl	seqIter(%rip), %eax
	cmpl	%eax, %edx
	jb	.L4
	movl	idx(%rip), %eax
	addl	$1, %eax
	movl	%eax, idx(%rip)
.L2:
	movl	idx(%rip), %edx
	movl	loopIter(%rip), %eax
	cmpl	%eax, %edx
	jb	.L5
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	wrapper, .-wrapper
	.ident	"GCC: (Ubuntu/Linaro 4.6.3-1ubuntu5) 4.6.3"
	.section	.note.GNU-stack,"",@progbits
